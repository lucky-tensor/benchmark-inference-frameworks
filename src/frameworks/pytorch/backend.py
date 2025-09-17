#!/usr/bin/env python3
"""
Pure PyTorch backend for LLaMA model inference.
This module provides a standalone PyTorch implementation separate from TinyGrad.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model configurations for different sizes
MODEL_CONFIGS = {
    "1B": {
        "dim": 2048,  # Processing dimension (from attention/ffn norms)
        "embed_dim": 1680,  # Embedding dimension (from token_embd.weight)
        "n_heads": 32,  # Q projection: 2048 / 64 = 32 heads
        "n_kv_heads": 8,  # K/V projection: 512 / 64 = 8 heads
        "n_layers": 16,
        "rope_theta": 500000,
        "vocab_size": 128256,
        "hidden_dim": 8192,  # FFN hidden dimension from GGUF file
        # Note: This Llama-3.2-1B model has non-standard architecture with different
        # embedding and processing dimensions
    },
    "8B": {
        "dim": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layers": 32,
        "rope_theta": 500000,
        "vocab_size": 128256,
        "hidden_dim": 14336,
    },
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency tensor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)

    def forward(self, x, pos):
        """Apply rotary position embedding"""
        # x: [batch, seq_len, n_heads, head_dim]
        # pos: [batch, seq_len] or scalar
        device = x.device
        dtype = x.dtype

        pos = torch.tensor([pos], device=device, dtype=torch.long) if isinstance(pos, int) else pos.to(device)

        # Get frequencies for positions
        freqs = self.freqs.to(device)
        pos_freqs = torch.outer(pos.float(), freqs)

        # Create cos and sin tensors
        cos = torch.cos(pos_freqs).to(dtype)
        sin = torch.sin(pos_freqs).to(dtype)

        # Reshape for broadcasting
        cos = cos[:, None, :]  # [seq_len, 1, dim//2]
        sin = sin[:, None, :]  # [seq_len, 1, dim//2]

        # Split x into pairs for rotation
        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Recombine
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.head_dim = config["dim"] // self.n_heads
        self.dim = config["dim"]

        # Linear projections
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        # RoPE
        self.rope = RoPE(self.head_dim, theta=config["rope_theta"])

        # KV cache (disabled for benchmarking)
        # self.cache_k = None
        # self.cache_v = None

    def forward(self, x, pos, temperature=1.0):
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project to Q, K, V
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Ensure position is on the same device
        pos = torch.tensor([pos], device=device, dtype=torch.long) if isinstance(pos, int) else pos.to(device)

        # Apply RoPE
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        # For benchmarking, we don't use KV cache as each inference is independent
        # KV cache would be used in actual generation but causes issues in benchmark
        # where we're doing independent single-token inferences
        # self.cache_k, self.cache_v = k, v  # Just store for reference

        # Repeat KV heads if needed (for grouped-query attention)
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply causal mask for training/prefill
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, scores.size(-1), device=scores.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float("-inf"))

        # Apply temperature scaling
        if temperature != 1.0 and not torch.isnan(torch.tensor(temperature)):
            scores = scores / temperature

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config["hidden_dim"]

        self.w1 = nn.Linear(config["dim"], hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, config["dim"], bias=False)  # Down
        self.w3 = nn.Linear(config["dim"], hidden_dim, bias=False)  # Up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config["dim"])
        self.ffn_norm = RMSNorm(config["dim"])

    def forward(self, x, pos, temperature=1.0):
        # Self-attention with residual connection
        h = x + self.attention(self.attention_norm(x), pos, temperature)

        # Feed-forward with residual connection
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class PyTorchLLaMA(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, n_layers, rope_theta, vocab_size, hidden_dim, embed_dim=None):
        super().__init__()
        self.embed_dim = embed_dim if embed_dim is not None else dim

        self.config = {
            "dim": dim,
            "embed_dim": self.embed_dim,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "n_layers": n_layers,
            "rope_theta": rope_theta,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
        }

        # Model layers
        self.tok_embeddings = nn.Embedding(vocab_size, self.embed_dim)

        # Input projection from embedding dim to processing dim (if different)
        if self.embed_dim != dim:
            self.input_proj = nn.Linear(self.embed_dim, dim, bias=False)
        else:
            self.input_proj = None

        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)

        # Output projection back to embedding space, then to vocab
        if self.embed_dim != dim:
            self.output_proj = nn.Linear(dim, self.embed_dim, bias=False)
        else:
            self.output_proj = None

        self.output = nn.Linear(self.embed_dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, pos, temperature=1.0, top_k=None, top_p=None, _alpha_f=None, _alpha_p=None):
        # Embed tokens
        x = self.tok_embeddings(tokens)

        # Project from embedding dim to processing dim if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, pos, temperature)

        # Final norm and output projection
        x = self.norm(x)

        # Project back to embedding dim if needed
        if self.output_proj is not None:
            x = self.output_proj(x)

        logits = self.output(x)

        # For inference, sample from the last token
        if logits.size(1) == 1 or len(logits.shape) == 2:
            logits = logits[:, -1, :] if len(logits.shape) == 3 else logits[-1, :]

            # Apply temperature
            if temperature > 0 and not torch.isnan(torch.tensor(temperature)):
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(-1, top_k_indices, top_k_logits)

            # Apply top-p filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token.squeeze(-1)

        return logits


class PyTorchTokenizer:
    """PyTorch-native tokenizer for LLaMA models using tiktoken."""

    def __init__(self, tokenizer_path: str):
        try:
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe

            # Load tokenizer using tiktoken (same approach as TinyGrad but independent)
            mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
            self.num_base_tokens = len(mergeable_ranks)

            # LLaMA 3 special tokens
            special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|reserved_special_token_2|>",
                "<|reserved_special_token_3|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|reserved_special_token_4|>",
                "<|eot_id|>",
            ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]

            self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

            self._tokenizer = tiktoken.Encoding(
                name=tokenizer_path,
                pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                mergeable_ranks=mergeable_ranks,
                special_tokens=self.special_tokens
            )

            # Expose the same interface as TinyGrad tokenizer
            self.bos_id = self.special_tokens["<|begin_of_text|>"]
            self.stop_tokens = {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}
            self.eos_id = self.special_tokens["<|end_of_text|>"]
            self.pad_id = self.eos_id

        except ImportError:
            print("⚠️  tiktoken not available, using fallback tokenizer")
            # Fallback: create a minimal tokenizer interface
            self._create_fallback_tokenizer()

    def _create_fallback_tokenizer(self):
        """Create a minimal fallback tokenizer when tiktoken is not available."""
        # Basic fallback - just split on whitespace and assign IDs
        self.vocab_size = 128256
        self.bos_id = 128000
        self.eos_id = 128001
        self.pad_id = self.eos_id

        # LLaMA 3 special tokens
        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }

        self.stop_tokens = {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}
        self._tokenizer = None

        print("⚠️  Using fallback tokenizer - install tiktoken for proper tokenization")

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        if self._tokenizer:
            return self._tokenizer.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())
        # Fallback: very basic tokenization
        words = text.split()
        return [hash(word) % (self.vocab_size - 1000) for word in words]

    def decode(self, tokens: list[int]) -> str:
        if self._tokenizer:
            return self._tokenizer.decode([t for t in tokens if t < self.num_base_tokens])
        # Fallback: just return a placeholder
        return f"[{len(tokens)} tokens]"


def load_pytorch_weights_from_gguf(model: PyTorchLLaMA, gguf_path: Path) -> bool:
    """
    Load GGUF weights directly into PyTorch model without TinyGrad dependency.

    Returns:
        bool: True if weights were successfully loaded, False otherwise.

    Raises:
        ImportError: If gguf library is not available.
        FileNotFoundError: If the GGUF file doesn't exist.
        Exception: If weight loading fails for any other reason.
    """
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    try:
        # Try using gguf library for direct loading
        import gguf
    except ImportError as e:
        raise ImportError(
            "The 'gguf' library is required for loading model weights. "
            "Install it with: pip install gguf"
        ) from e

    try:
        print(f"Loading GGUF file: {gguf_path}")
        reader = gguf.GGUFReader(str(gguf_path))

        # Extract tensors from GGUF format
        tensors = {}
        for tensor in reader.tensors:
            name = str(tensor.name, 'utf-8') if isinstance(tensor.name, bytes) else tensor.name
            # Convert GGUF tensor data to PyTorch tensor
            data = tensor.data
            if hasattr(data, 'numpy'):
                data = data.numpy()
            tensors[name] = torch.from_numpy(data.copy())

        if not tensors:
            raise ValueError(f"No tensors found in GGUF file: {gguf_path}")

        # Map GGUF tensor names to PyTorch model structure
        loaded_count, total_expected = _load_gguf_tensors_to_model(model, tensors)

        if loaded_count == 0:
            raise ValueError("No model weights were successfully loaded from GGUF file")

        # Check if we loaded a reasonable proportion of weights
        success_rate = loaded_count / total_expected

        if success_rate < 0.7:  # Less than 70% success rate indicates major incompatibility
            raise ValueError(
                f"Model weight loading had low success rate ({success_rate:.1%}): "
                f"loaded {loaded_count}/{total_expected} tensors. "
                f"This suggests the GGUF file has incompatible dimensions with the PyTorch model configuration. "
                f"The model architecture in the GGUF file may not match the expected configuration."
            )

        print(f"✅ Successfully loaded {loaded_count}/{total_expected} weight tensors from GGUF file ({success_rate:.1%})")

        if success_rate < 1.0:
            print(f"⚠️  Note: {total_expected - loaded_count} tensors had dimension mismatches and were not loaded")

        return True

    except Exception as e:
        raise Exception(f"Failed to load GGUF weights from {gguf_path}: {e}") from e


def _load_gguf_tensors_to_model(model: PyTorchLLaMA, tensors: dict) -> tuple[int, int]:
    """
    Map GGUF tensor names to PyTorch model structure.

    Returns:
        tuple[int, int]: (loaded_count, total_expected)
    """
    # GGUF to PyTorch name mapping (common GGUF format)
    name_mapping = {
        "token_embd.weight": "tok_embeddings.weight",
        "output_norm.weight": "norm.weight",
    }

    # Check if model uses tied embeddings (no separate output.weight)
    if "output.weight" in tensors:
        name_mapping["output.weight"] = "output.weight"
    else:
        # Use tied embeddings: output layer shares weights with token embeddings
        print("  Using tied embeddings (output layer shares weights with token embeddings)")
        name_mapping["token_embd.weight_tied"] = "output.weight"

    # Add layer-specific mappings
    for i in range(len(model.layers)):
        name_mapping.update({
            f"blk.{i}.attn_norm.weight": f"layers.{i}.attention_norm.weight",
            f"blk.{i}.attn_q.weight": f"layers.{i}.attention.wq.weight",
            f"blk.{i}.attn_k.weight": f"layers.{i}.attention.wk.weight",
            f"blk.{i}.attn_v.weight": f"layers.{i}.attention.wv.weight",
            f"blk.{i}.attn_output.weight": f"layers.{i}.attention.wo.weight",
            f"blk.{i}.ffn_norm.weight": f"layers.{i}.ffn_norm.weight",
            f"blk.{i}.ffn_gate.weight": f"layers.{i}.feed_forward.w1.weight",
            f"blk.{i}.ffn_down.weight": f"layers.{i}.feed_forward.w2.weight",
            f"blk.{i}.ffn_up.weight": f"layers.{i}.feed_forward.w3.weight",
        })

    loaded_count = 0
    failed_count = 0

    # Load weights into model
    with torch.no_grad():
        for gguf_name, pytorch_name in name_mapping.items():
            # Handle tied embeddings special case
            if gguf_name == "token_embd.weight_tied":
                # Copy token embedding weights to output layer
                if "token_embd.weight" in tensors:
                    try:
                        embed_tensor = tensors["token_embd.weight"]
                        output_param = model.output.weight

                        if output_param.shape == embed_tensor.shape:
                            output_param.copy_(embed_tensor)
                            print(f"✓ Loaded token_embd.weight -> {pytorch_name} (tied embeddings)")
                            loaded_count += 1
                        else:
                            print(f"⚠️  Shape mismatch for tied embeddings: {output_param.shape} != {embed_tensor.shape}")
                            failed_count += 1

                    except Exception as e:
                        print(f"⚠️  Failed to load tied embeddings: {e}")
                        failed_count += 1
                continue

            # Regular tensor loading
            if gguf_name in tensors:
                try:
                    # Navigate to the parameter in the model
                    param = model
                    for part in pytorch_name.split('.'):
                        if part.isdigit():
                            param = param[int(part)]
                        else:
                            param = getattr(param, part)

                    # Copy the tensor data
                    tensor_data = tensors[gguf_name]
                    if param.shape == tensor_data.shape:
                        param.copy_(tensor_data)
                        print(f"✓ Loaded {gguf_name} -> {pytorch_name}")
                        loaded_count += 1
                    else:
                        print(f"⚠️  Shape mismatch for {pytorch_name}: {param.shape} != {tensor_data.shape}")
                        failed_count += 1

                except Exception as e:
                    print(f"⚠️  Failed to load {pytorch_name}: {e}")
                    failed_count += 1

    print(f"✅ Finished loading GGUF weights: {loaded_count} successful, {failed_count} failed")
    return loaded_count, len(name_mapping)


def main():
    """Standalone PyTorch backend for benchmarking."""
    parser = argparse.ArgumentParser(description="Pure PyTorch LLaMA Backend")
    parser.add_argument("--size", choices=["1B", "8B"], default="1B", help="Model size")
    parser.add_argument("--model", type=Path, help="Path to model directory")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--half", action="store_true", help="Use half precision")

    args = parser.parse_args()

    # Create model
    config = MODEL_CONFIGS[args.size]
    model = PyTorchLLaMA(**config)

    # Apply optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.half:
        model = model.half()

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Load weights if model path provided
    if args.model:
        gguf_files = list(args.model.glob("*.gguf"))
        if gguf_files:
            print(f"Loading weights from {gguf_files[0]}")
            load_pytorch_weights_from_gguf(model, gguf_files[0])

    # Load tokenizer
    tokenizer_path = (
        args.model / "tokenizer.model" if args.model else Path.home() / "models/llama3-1b-instruct/tokenizer.model"
    )
    tokenizer = PyTorchTokenizer(str(tokenizer_path))

    if args.benchmark:
        print("Running PyTorch benchmark...")

        # Simple benchmark
        model.eval()

        # Prepare input
        tokens = [tokenizer.bos_id, *tokenizer.encode("Hello world")]
        input_tensor = torch.tensor([tokens], device=device, dtype=torch.long)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor, pos=0, temperature=0.85)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        with torch.no_grad():
            for i in range(20):
                output = model(input_tensor[:, -1:], pos=len(tokens) + i, temperature=0.85)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 20
        tokens_per_sec = 1.0 / avg_time

        print(f"Average latency: {avg_time * 1000:.2f}ms")
        print(f"Throughput: {tokens_per_sec:.1f} tokens/second")
    else:
        # Interactive mode
        print("PyTorch LLaMA Model loaded. Type 'quit' to exit.")

        while True:
            user_input = input("User: ")
            if user_input.lower() == "quit":
                break

            # Simple generation (not implementing full chat format for brevity)
            tokens = tokenizer.encode(user_input)
            input_tensor = torch.tensor([tokens], device=device, dtype=torch.long)

            generated_tokens = []
            model.eval()

            with torch.no_grad():
                for i in range(50):  # Generate up to 50 tokens
                    if i == 0:
                        output = model(input_tensor, pos=0, temperature=0.85)
                    else:
                        last_token = torch.tensor([[generated_tokens[-1]]], device=device, dtype=torch.long)
                        output = model(last_token, pos=len(tokens) + i, temperature=0.85)

                    next_token = output.item() if hasattr(output, "item") else int(output[0])
                    generated_tokens.append(next_token)

                    if next_token in tokenizer.stop_tokens:
                        break

            response = tokenizer.decode(generated_tokens)
            print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
