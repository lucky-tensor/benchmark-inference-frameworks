#!/usr/bin/env python3
"""
Pure PyTorch backend for LLaMA model inference.
This module provides a standalone PyTorch implementation separate from TinyGrad.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model configurations for different sizes
MODEL_CONFIGS = {
    "1B": {
        "dim": 2048,
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layers": 16,
        "rope_theta": 500000,
        "vocab_size": 128256,
        "hidden_dim": 8192,
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
    def __init__(self, dim, n_heads, n_kv_heads, n_layers, rope_theta, vocab_size, hidden_dim):
        super().__init__()
        self.config = {
            "dim": dim,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "n_layers": n_layers,
            "rope_theta": rope_theta,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
        }

        # Model layers
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, pos, temperature=1.0, top_k=None, top_p=None, _alpha_f=None, _alpha_p=None):
        # Embed tokens
        x = self.tok_embeddings(tokens)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, pos, temperature)

        # Final norm and output projection
        x = self.norm(x)
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
    """Simple tokenizer interface compatible with TinyGrad tokenizer."""

    def __init__(self, tokenizer_path: str):
        # Import TinyGrad tokenizer for compatibility
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from common.tokenizer import Tokenizer

        self._tokenizer = Tokenizer(tokenizer_path)

        # Expose the same interface
        self.bos_id = self._tokenizer.bos_id
        self.stop_tokens = self._tokenizer.stop_tokens
        # Use the first stop token as eos_id for compatibility
        self.eos_id = next(iter(self.stop_tokens))
        self.pad_id = getattr(self._tokenizer, "pad_id", self.eos_id)
        self.special_tokens = self._tokenizer.special_tokens

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)


def load_pytorch_weights_from_tinygrad(model: PyTorchLLaMA, gguf_path: Path):
    """Load GGUF weights into PyTorch model by converting from TinyGrad format."""
    from pathlib import Path

    # Import TinyGrad modules from new location
    sys.path.insert(0, str(Path(__file__).parent.parent / "frameworks" / "tinygrad"))
    from llama.model_config import build_transformer

    # Load TinyGrad model
    tinygrad_model = build_transformer(gguf_path, model_size="1B", quantize=None, device="cpu")

    # Convert weights
    with torch.no_grad():
        # Token embeddings
        model.tok_embeddings.weight.copy_(torch.tensor(tinygrad_model.tok_embeddings.weight.numpy()))

        # Layers
        for i, (pytorch_layer, tinygrad_layer) in enumerate(zip(model.layers, tinygrad_model.layers, strict=False)):
            # Attention weights
            pytorch_layer.attention.wq.weight.copy_(torch.tensor(tinygrad_layer.attention.wq.weight.numpy()))
            pytorch_layer.attention.wk.weight.copy_(torch.tensor(tinygrad_layer.attention.wk.weight.numpy()))
            pytorch_layer.attention.wv.weight.copy_(torch.tensor(tinygrad_layer.attention.wv.weight.numpy()))
            pytorch_layer.attention.wo.weight.copy_(torch.tensor(tinygrad_layer.attention.wo.weight.numpy()))

            # Feed forward weights
            pytorch_layer.feed_forward.w1.weight.copy_(torch.tensor(tinygrad_layer.feed_forward.w1.weight.numpy()))
            pytorch_layer.feed_forward.w2.weight.copy_(torch.tensor(tinygrad_layer.feed_forward.w2.weight.numpy()))
            pytorch_layer.feed_forward.w3.weight.copy_(torch.tensor(tinygrad_layer.feed_forward.w3.weight.numpy()))

            # Layer norms
            pytorch_layer.attention_norm.weight.copy_(torch.tensor(tinygrad_layer.attention_norm.weight.numpy()))
            pytorch_layer.ffn_norm.weight.copy_(torch.tensor(tinygrad_layer.ffn_norm.weight.numpy()))

        # Final norm and output
        model.norm.weight.copy_(torch.tensor(tinygrad_model.norm.weight.numpy()))
        model.output.weight.copy_(torch.tensor(tinygrad_model.output.weight.numpy()))


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
            load_pytorch_weights_from_tinygrad(model, gguf_files[0])

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
