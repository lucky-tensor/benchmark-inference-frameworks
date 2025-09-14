#!/usr/bin/env python3
"""
PyTorch Pure Implementation for LLaMA 3 - Baseline for comparison.

This file implements the LLaMA 3 architecture using pure PyTorch without
tinygrad kernel optimizations. Used as baseline for comparing against
pytorch-opt-tinygrad.py which uses tinygrad kernel acceleration.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# RoPE (Rotary Position Embedding) Implementation
# ============================================================================


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device="cuda") -> torch.Tensor:
    """Precompute the frequency tensor for rotary embeddings."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this implementation")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)] / dim))
    freqs = torch.arange(end, device=device).unsqueeze(1) * freqs.unsqueeze(0)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1).view(1, end, 1, dim // 2, 2)


def complex_mult(A: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Complex multiplication: (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)."""
    a, b = A[..., 0:1], A[..., 1:2]
    ro = a * c - b * d
    co = a * d + b * c
    return torch.cat([ro, co], dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], (
        f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
    )

    xq = xq.reshape(*xq.shape[0:-1], -1, 2)
    xk = xk.reshape(*xk.shape[0:-1], -1, 2)

    c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
    xq_out = complex_mult(xq, c, d)
    xk_out = complex_mult(xk, c, d)

    return xq_out.flatten(-2), xk_out.flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key-value heads for multi-query attention."""
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x.repeat(1, 1, 1, n_rep).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)


# ============================================================================
# RMS Normalization
# ============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ============================================================================
# Attention Module
# ============================================================================


class Attention(nn.Module):
    """Multi-head attention with KV cache support."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int | None = None, max_context: int = 0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.max_context = max_context

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # KV cache will be initialized lazily
        self.register_buffer("cache_kv", None, persistent=False)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Linear projections
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV caching
        if self.max_context > 0:
            if self.cache_kv is None:
                self.cache_kv = torch.zeros(
                    2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype, device=x.device
                )

            # Update cache
            self.cache_kv[0, :, start_pos : start_pos + seqlen] = xk
            self.cache_kv[1, :, start_pos : start_pos + seqlen] = xv

            keys = self.cache_kv[0, :, : start_pos + seqlen]
            values = self.cache_kv[1, :, : start_pos + seqlen]
        else:
            keys, values = xk, xv

        # Repeat KV heads for multi-query attention
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Reshape for attention
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(attn)


# ============================================================================
# Feed Forward Module
# ============================================================================


class FeedForward(nn.Module):
    """SwiGLU feed forward network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # output projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# Transformer Block
# ============================================================================


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed forward."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        max_context: int,
    ):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, max_context)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))


# ============================================================================
# PyTorch LLaMA Model
# ============================================================================


class PyTorchLLaMA(nn.Module):
    """Complete LLaMA model implemented in PyTorch."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        norm_eps: float,
        vocab_size: int,
        n_kv_heads: int | None = None,
        rope_theta: float = 10000.0,
        max_context: int = 1024,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.vocab_size = vocab_size
        self.max_context = max_context

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, hidden_dim, n_heads, self.n_kv_heads, norm_eps, max_context)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim, norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Precompute rotary embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim // n_heads, max_context * 2, rope_theta),
            persistent=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        temperature: float = 0.85,
    ) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Get rotary embeddings for current sequence
        freqs_cis = self.freqs_cis[:, start_pos : start_pos + seqlen].to(h.dtype)

        # Create causal mask if needed
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, start_pos + seqlen), float("-inf"), dtype=h.dtype, device=h.device).triu(
                start_pos + 1
            )
            mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        logits = self.output(h)

        # Return logits if temperature is NaN (for logits mode)
        if math.isnan(temperature):
            return logits

        # Apply sampling
        logits = logits[:, -1, :] / temperature  # Get last position and apply temperature

        # Simple sampling (can be enhanced with top_k, top_p later)
        if temperature < 1e-6:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# ============================================================================
# Tokenizer Interface
# ============================================================================


class PyTorchTokenizer:
    """Wrapper around tinygrad tokenizer for PyTorch compatibility."""

    def __init__(self, tokenizer_path: str):
        # Import tinygrad tokenizer
        from common.tokenizer import Tokenizer

        self.tokenizer = Tokenizer(tokenizer_path)
        self.bos_id = self.tokenizer.bos_id
        self.special_tokens = self.tokenizer.special_tokens
        self.stop_tokens = [
            self.tokenizer.special_tokens["<|end_of_text|>"],
            self.tokenizer.special_tokens["<|eot_id|>"],
        ]

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        return self.tokenizer.encode(text, allow_special=allow_special)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


# ============================================================================
# Weight Loading from TinyGrad Format
# ============================================================================


def load_pytorch_weights_from_tinygrad(model: PyTorchLLaMA, weight_path: Path) -> None:
    """Load weights from tinygrad GGUF format into PyTorch model."""
    # Import tinygrad components
    from llama.model_config import load_weights

    print(f"Loading weights from {weight_path}...")
    tinygrad_weights = load_weights(str(weight_path))

    # Convert tinygrad weights to PyTorch format with proper name mapping
    state_dict = {}

    for tg_name, tinygrad_tensor in tinygrad_weights.items():
        # Convert tinygrad tensor to PyTorch tensor
        if hasattr(tinygrad_tensor, "numpy"):
            numpy_array = tinygrad_tensor.numpy()
        else:
            numpy_array = tinygrad_tensor.realize().numpy()

        pytorch_tensor = torch.from_numpy(numpy_array)

        # Map GGUF weight names to PyTorch model structure
        if tg_name.startswith("blk."):
            # Extract layer number and component
            parts = tg_name.split(".")
            layer_num = parts[1]
            component = ".".join(parts[2:])

            # Map component names
            if component == "attn_q.weight":
                pt_name = f"layers.{layer_num}.attention.wq.weight"
            elif component == "attn_k.weight":
                pt_name = f"layers.{layer_num}.attention.wk.weight"
            elif component == "attn_v.weight":
                pt_name = f"layers.{layer_num}.attention.wv.weight"
            elif component == "attn_output.weight":
                pt_name = f"layers.{layer_num}.attention.wo.weight"
            elif component == "attn_norm.weight":
                pt_name = f"layers.{layer_num}.attention_norm.weight"
            elif component == "ffn_gate.weight":
                pt_name = f"layers.{layer_num}.feed_forward.w1.weight"
            elif component == "ffn_down.weight":
                pt_name = f"layers.{layer_num}.feed_forward.w2.weight"
            elif component == "ffn_up.weight":
                pt_name = f"layers.{layer_num}.feed_forward.w3.weight"
            elif component == "ffn_norm.weight":
                pt_name = f"layers.{layer_num}.ffn_norm.weight"
            else:
                print(f"Unknown layer component: {component}")
                continue

        elif tg_name == "token_embd.weight":
            pt_name = "tok_embeddings.weight"
        elif tg_name == "output_norm.weight":
            pt_name = "norm.weight"
        elif tg_name == "output.weight":
            pt_name = "output.weight"
        elif tg_name == "rope_freqs.weight":
            # Skip rope frequencies - they are computed, not loaded
            continue
        else:
            print(f"Unknown weight: {tg_name}")
            continue

        state_dict[pt_name] = pytorch_tensor

    # Load the state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Missing keys in PyTorch model: {len(missing_keys)} keys")
        print(f"First few: {missing_keys[:3]}")
    if unexpected_keys:
        print(f"Unexpected keys in weights: {len(unexpected_keys)} keys")
        print(f"First few: {unexpected_keys[:3]}")

    loaded_keys = len(state_dict)
    total_keys = len(model.state_dict())
    print(f"Successfully loaded {loaded_keys}/{total_keys} weights")

    print("Weights loaded successfully!")


# ============================================================================
# Benchmarking Functions
# ============================================================================


def run_pytorch_benchmark(model: PyTorchLLaMA, tokenizer: PyTorchTokenizer, args) -> None:
    """Run PyTorch benchmark matching tinygrad benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply same precision as TinyGrad for fair comparison
    if hasattr(args, "use_half") and args.use_half:
        model = model.half()  # Use float16 like TinyGrad mixed precision

    model = model.to(device)

    # Apply torch.compile for JIT optimization to match TinyGrad's JIT
    if hasattr(args, "use_compile") and args.use_compile and hasattr(torch, "compile"):
        print("Applying torch.compile for fair comparison with TinyGrad JIT...")
        model = torch.compile(model)

    model.eval()

    # Prepare benchmark input (matching tinygrad benchmark)
    def encode_message(role: str, content: str) -> list[int]:
        if role == "user":
            return [
                tokenizer.special_tokens["<|start_header_id|>"],
                *tokenizer.encode("user"),
                tokenizer.special_tokens["<|end_header_id|>"],
                *tokenizer.encode("\n\n" + content.strip()),
                tokenizer.special_tokens["<|eot_id|>"],
            ]
        return []

    def encode_role(role: str) -> list[int]:
        return [
            tokenizer.special_tokens["<|start_header_id|>"],
            *tokenizer.encode(role),
            tokenizer.special_tokens["<|end_header_id|>"],
            *tokenizer.encode("\n\n"),
        ]

    toks = [tokenizer.bos_id, *encode_message("user", "Hello."), *encode_role("assistant")]

    # Prefill (similar to tinygrad prefill)
    with torch.no_grad():
        prefill_tokens = torch.tensor([toks[:-1]], device=device, dtype=torch.long)
        _ = model(prefill_tokens, 0, temperature=float("nan"))  # Just run forward pass

    start_pos = len(toks) - 1
    last_tok = toks[-1]

    print("Running PyTorch benchmark (20 iterations)...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    times = []
    total_memory = 0

    with torch.no_grad():
        for i in range(20):
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.perf_counter()

            input_tensor = torch.tensor([[last_tok]], device=device, dtype=torch.long)
            tok = model(input_tensor, start_pos, temperature=0.85)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.perf_counter()

            iteration_time = end_time - start_time
            times.append(iteration_time)

            if device.type == "cuda":
                memory_used = torch.cuda.max_memory_allocated() / 1e9  # GB
                total_memory = max(total_memory, memory_used)

            token_speed = 1.0 / iteration_time

            print(f"Iteration {i + 1:2d}: {iteration_time * 1000:6.2f}ms, {token_speed:6.1f} tok/s")

            start_pos += 1
            last_tok = tok.item()

    # Statistics
    avg_time = sum(times) / len(times)
    avg_speed = 1.0 / avg_time

    print("\nPyTorch Benchmark Results:")
    print(f"Average time per token: {avg_time * 1000:.2f}ms")
    print(f"Average tokens per second: {avg_speed:.1f} tok/s")
    print(f"Peak memory usage: {total_memory:.2f}GB")


# ============================================================================
# Model Configuration Mapping
# ============================================================================

MODEL_CONFIGS = {
    "1B": {
        "dim": 2048,
        "hidden_dim": 8192,
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layers": 16,
        "norm_eps": 1e-5,
        "rope_theta": 500000,
        "vocab_size": 128256,
        "max_context": 8192,
    },
    "8B": {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layers": 32,
        "norm_eps": 1e-5,
        "rope_theta": 500000,
        "vocab_size": 128256,
        "max_context": 8192,
    },
}


# ============================================================================
# Main CLI Interface
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="PyTorch LLaMA 3 Benchmark")
    parser.add_argument("--model", type=Path, help="Model weights path")
    parser.add_argument("--size", choices=["1B", "8B"], default="1B", help="Model size")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")

    args = parser.parse_args()

    # Get model config
    config = MODEL_CONFIGS[args.size]

    # Create model
    model = PyTorchLLaMA(**config)
    print(f"Created PyTorch LLaMA {args.size} model")

    # Load tokenizer
    if args.model and args.model.is_dir():
        tokenizer_path = args.model / "tokenizer.model"
    elif args.model:
        tokenizer_path = args.model.parent / "tokenizer.model"
    else:
        # Try default path
        tokenizer_path = Path.home() / "models" / f"llama3-{args.size.lower()}-instruct" / "tokenizer.model"

    tokenizer = PyTorchTokenizer(str(tokenizer_path))
    print(f"Loaded tokenizer from {tokenizer_path}")

    # Load weights if provided
    if args.model:
        # If model is a directory, look for GGUF file
        if args.model.is_dir():
            # Look for GGUF file in directory
            gguf_files = list(args.model.glob("*.gguf"))
            if gguf_files:
                weight_path = gguf_files[0]  # Use first GGUF file found
            else:
                print(f"No GGUF files found in {args.model}")
                weight_path = None
        else:
            weight_path = args.model

        if weight_path:
            load_pytorch_weights_from_tinygrad(model, weight_path)

    # Run benchmark
    if args.benchmark:
        run_pytorch_benchmark(model, tokenizer, args)
    else:
        print("Use --benchmark flag to run performance comparison")


if __name__ == "__main__":
    main()
