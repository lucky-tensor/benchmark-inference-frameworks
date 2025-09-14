"""Common Inference Engine

Shared inference logic that can be used by different frameworks and interfaces.
This eliminates code duplication between Q&A mode and benchmarking.
"""

import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinygrad import GlobalCounters
from tinygrad.helpers import DEBUG, Profiling, Timing


@dataclass
class InferenceStats:
    """Statistics for inference performance."""

    start_time: float = 0.0
    first_token_time: float | None = None
    tokens_generated: int = 0
    total_time: float = 0.0

    def start_generation(self) -> None:
        """Start timing generation."""
        self.start_time = time.time()

    def record_first_token(self) -> None:
        """Record first token generation time."""
        if self.first_token_time is None:
            self.first_token_time = time.time() - self.start_time

    def record_token(self) -> None:
        """Record a token generation."""
        self.tokens_generated += 1

    def finalize(self) -> None:
        """Finalize timing measurements."""
        self.total_time = time.time() - self.start_time

    def get_tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.total_time > 0:
            return self.tokens_generated / self.total_time
        return 0.0

    def format_stats(self) -> str:
        """Format statistics for display."""
        tokens_per_sec = self.get_tokens_per_second()
        first_token_ms = (self.first_token_time * 1000) if self.first_token_time else 0.0

        return (
            f"Generated {self.tokens_generated} tokens in {self.total_time:.2f}s "
            f"({tokens_per_sec:.1f} tok/s, first token: {first_token_ms:.1f}ms)"
        )


class InferenceEngine:
    """Common inference engine that supports multiple frameworks."""

    def __init__(self, framework: str, model: Any, tokenizer: Any, device: Any, param_bytes: int, args: Any):
        """Initialize inference engine.

        Args:
            framework: Framework name ('tinygrad', 'pytorch', 'hybrid')
            model: The model instance
            tokenizer: Tokenizer instance
            device: Device configuration
            param_bytes: Number of parameter bytes
            args: Command line arguments with timing/profile flags
        """
        self.framework = framework
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.param_bytes = param_bytes
        self.args = args

    def generate_tokens(
        self, input_tokens: list[int], max_tokens: int = 100, chat_interface: Any | None = None
    ) -> Generator[dict[str, Any], None, None]:
        """Generate tokens from input with timing and profiling.

        Args:
            input_tokens: Input token sequence
            max_tokens: Maximum tokens to generate
            chat_interface: Optional chat interface for stop token checking

        Yields:
            Dict with token info and timing stats
        """
        if self.framework == "tinygrad":
            yield from self._generate_tinygrad(input_tokens, max_tokens, chat_interface)
        elif self.framework == "pytorch":
            yield from self._generate_pytorch(input_tokens, max_tokens, chat_interface)
        elif self.framework == "hybrid":
            yield from self._generate_hybrid(input_tokens, max_tokens, chat_interface)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def _generate_tinygrad(
        self, input_tokens: list[int], max_tokens: int, chat_interface: Any | None
    ) -> Generator[dict[str, Any], None, None]:
        """Generate using TinyGrad."""
        from tinygrad import Tensor

        from common import generation
        from common.generation import prefill

        # Initialize position
        start_pos = prefill(self.model, input_tokens)
        last_tok = input_tokens[-1]

        for i in range(max_tokens):
            # Reset counters for timing
            GlobalCounters.reset()
            timing_enabled = self.args.timing or self.args.profile

            if timing_enabled:
                print("")
            st = GlobalCounters.time_sum_s

            # Generate next token with profiling/timing
            with (
                Profiling(enabled=self.args.profile),
                Timing(
                    "total ",
                    enabled=self.args.timing,
                    on_exit=lambda x: f", {1e9 / x:.2f} tok/s, {GlobalCounters.global_mem / x:.2f} GB/s, "
                    f"param {self.param_bytes / x:.2f} GB/s",
                ),
            ):
                with Timing(
                    "enqueue in ",
                    enabled=self.args.timing,
                    on_exit=(
                        lambda _et, start_time=st: (
                            f", {(GlobalCounters.time_sum_s - start_time) * 1e3:.2f} ms on {self.device}"
                            if DEBUG >= 2
                            else ""
                        )
                        + f", {GlobalCounters.global_ops * 1e-9:.2f} GOPS, "
                        f"{GlobalCounters.global_mem * 1e-9:.2f} GB"
                        + (
                            f",{GlobalCounters.global_mem * 1e-9 / (GlobalCounters.time_sum_s - start_time):.2f} GB/s,"
                            f"param {self.param_bytes * 1e-9 / (GlobalCounters.time_sum_s - start_time):.2f} GB/s"
                            if DEBUG >= 2
                            else ""
                        )
                    )
                    if DEBUG
                    else None,
                ):
                    tok = self.model(
                        Tensor([[last_tok]], device=self.device),
                        start_pos,
                        generation.TEMPERATURE,
                        generation.TOP_K,
                        generation.TOP_P,
                        generation.ALPHA_F,
                        generation.ALPHA_P,
                    )
                tok = tok.item()

            # Check for stop tokens
            if chat_interface and tok in chat_interface.get_stop_tokens():
                break

            # Yield token with metadata
            yield {
                "token": tok,
                "position": start_pos,
                "is_first": i == 0,
                "timing_info": {
                    "global_ops": GlobalCounters.global_ops,
                    "global_mem": GlobalCounters.global_mem,
                    "time_sum": GlobalCounters.time_sum_s,
                },
            }

            start_pos += 1
            last_tok = tok

    def _generate_pytorch(
        self, _input_tokens: list[int], max_tokens: int, _chat_interface: Any | None
    ) -> Generator[dict[str, Any], None, None]:
        """Generate using PyTorch (placeholder)."""
        # This would implement PyTorch generation
        # For now, just yield dummy tokens
        for i in range(min(max_tokens, 5)):  # Limit for demo
            yield {
                "token": 1,  # Dummy token
                "position": i,
                "is_first": i == 0,
                "timing_info": {},
            }

    def _generate_hybrid(
        self, _input_tokens: list[int], max_tokens: int, _chat_interface: Any | None
    ) -> Generator[dict[str, Any], None, None]:
        """Generate using hybrid PyTorch-TinyGrad."""
        # This would use our accelerated implementation
        # For now, just yield dummy tokens
        for i in range(min(max_tokens, 5)):  # Limit for demo
            yield {
                "token": 1,  # Dummy token
                "position": i,
                "is_first": i == 0,
                "timing_info": {},
            }

    def run_interactive_session(self, system_prompt: str = "You are a helpful assistant.") -> None:
        """Run interactive Q&A session."""
        from common.chat_interface import ChatSession, LLaMA3ChatInterface, MessageRole
        from common.generation import prefill

        chat_interface = LLaMA3ChatInterface(self.tokenizer)
        session = ChatSession([], system_prompt)

        initial_tokens = chat_interface.encode_chat_session(session)
        start_pos = prefill(self.model, initial_tokens)

        print("Interactive Q&A mode. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input(chat_interface.format_interactive_prompt())
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                session.add_message(MessageRole.USER, user_input)
                context_tokens, _expected_role = chat_interface.prepare_generation_context(session)

                stats = InferenceStats()
                stats.start_generation()

                # Handle context extension
                if len(context_tokens) > len(initial_tokens):
                    new_tokens = context_tokens[len(initial_tokens) :]
                    start_pos = prefill(self.model, new_tokens, start_pos=start_pos)
                    initial_tokens = context_tokens

                # Generate response
                response_text = ""
                for token_info in self.generate_tokens(
                    context_tokens[-1:], max_tokens=256, chat_interface=chat_interface
                ):
                    token = token_info["token"]

                    if token_info["is_first"]:
                        stats.record_first_token()
                    stats.record_token()

                    decoded = chat_interface.decode_tokens([token])
                    response_text += decoded
                    print(decoded, end="", flush=True)

                stats.finalize()
                print(f"\n{stats.format_stats()}\n")

                session.add_message(MessageRole.ASSISTANT, response_text, stats)
                initial_tokens = chat_interface.encode_chat_session(session)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

    def run_single_inference(self, prompt: str, max_tokens: int = 100) -> str:
        """Run single inference and return result."""
        from common.chat_interface import ChatSession, LLaMA3ChatInterface, MessageRole

        chat_interface = LLaMA3ChatInterface(self.tokenizer)
        session = ChatSession([], "You are a helpful assistant.")
        session.add_message(MessageRole.USER, prompt)

        context_tokens, _expected_role = chat_interface.prepare_generation_context(session)

        stats = InferenceStats()
        stats.start_generation()

        response_text = ""
        for token_info in self.generate_tokens(
            context_tokens[-1:], max_tokens=max_tokens, chat_interface=chat_interface
        ):
            token = token_info["token"]

            if token_info["is_first"]:
                stats.record_first_token()
            stats.record_token()

            decoded = chat_interface.decode_tokens([token])
            response_text += decoded

        stats.finalize()
        print(f"{stats.format_stats()}")

        return response_text.strip()

    def run_benchmark(self, num_iterations: int = 5, prompts: list[str] | None = None) -> dict[str, Any]:
        """Run benchmark with multiple prompts and iterations."""
        if prompts is None:
            prompts = [
                "What is artificial intelligence?",
                "Explain quantum computing briefly.",
                "Write a short poem about technology.",
            ]

        print(f"🏁 Running {self.framework} benchmark ({num_iterations} iterations)")

        results = {
            "framework": self.framework,
            "iterations": num_iterations,
            "prompts": len(prompts),
            "generation_times": [],
            "tokens_per_second": [],
            "first_token_times": [],
        }

        for i in range(num_iterations):
            print(f"   Iteration {i + 1}/{num_iterations}")

            for _prompt in prompts:
                stats = InferenceStats()
                stats.start_generation()

                # Generate response (limit tokens for benchmarking)
                for token_info in self.generate_tokens([1], max_tokens=20):  # Short for speed
                    if token_info["is_first"]:
                        stats.record_first_token()
                    stats.record_token()

                stats.finalize()

                results["generation_times"].append(stats.total_time)
                results["tokens_per_second"].append(stats.get_tokens_per_second())
                if stats.first_token_time:
                    results["first_token_times"].append(stats.first_token_time)

        # Calculate averages
        if results["generation_times"]:
            results["avg_generation_time"] = sum(results["generation_times"]) / len(results["generation_times"])
            results["avg_tokens_per_second"] = sum(results["tokens_per_second"]) / len(results["tokens_per_second"])

        if results["first_token_times"]:
            results["avg_first_token_time"] = sum(results["first_token_times"]) / len(results["first_token_times"])

        return results


def create_inference_engine(
    framework: str,
    model_path: Path | None = None,
    model_size: str = "1B",
    quantize: str | None = None,
    device: str = "auto",
    args: Any = None,
) -> InferenceEngine:
    """Factory function to create appropriate inference engine.

    Args:
        framework: 'tinygrad', 'pytorch', or 'hybrid'
        model_path: Path to model weights
        model_size: Model size identifier
        quantize: Quantization method
        device: Device specification
        args: Command line arguments

    Returns:
        Configured InferenceEngine instance
    """
    if framework == "tinygrad":
        return _create_tinygrad_engine(model_path, model_size, quantize, device, args)
    if framework == "pytorch":
        return _create_pytorch_engine(model_path, model_size, quantize, device, args)
    if framework == "hybrid":
        return _create_hybrid_engine(model_path, model_size, quantize, device, args)
    raise ValueError(f"Unknown framework: {framework}")


def _create_tinygrad_engine(
    model_path: Path | None, model_size: str, quantize: str | None, _device: str, args: Any
) -> InferenceEngine:
    """Create TinyGrad inference engine."""
    from tinygrad import Device
    from tinygrad.nn.state import get_parameters

    from common.tokenizer import Tokenizer
    from llama.model_config import build_transformer, resolve_model_path

    # Resolve model path
    model_path = resolve_model_path(model_path, model_size, False)
    assert model_path is not None, "Could not resolve model path"

    # Setup tokenizer and model
    tokenizer = Tokenizer(str((model_path if model_path.is_dir() else model_path.parent) / "tokenizer.model"))
    device_config = (
        tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard if hasattr(args, "shard") else 1))
        if hasattr(args, "shard") and args.shard > 1
        else Device.DEFAULT
    )
    model = build_transformer(model_path, model_size=model_size, quantize=quantize, device=device_config)
    param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(model))

    return InferenceEngine("tinygrad", model, tokenizer, device_config, param_bytes, args)


def _create_pytorch_engine(
    _model_path: Path | None, _model_size: str, _quantize: str | None, device: str, args: Any
) -> InferenceEngine:
    """Create PyTorch inference engine (placeholder)."""
    # This would implement PyTorch model loading
    return InferenceEngine("pytorch", None, None, device, 0, args)


def _create_hybrid_engine(
    model_path: Path | None, _model_size: str, quantize: str | None, device: str, args: Any
) -> InferenceEngine:
    """Create hybrid PyTorch-TinyGrad inference engine."""
    from llama.accelerated_llama import AcceleratedLLaMA3

    if model_path is None:
        model_path = Path("/dev/null")  # Dummy for testing

    # Create hybrid model
    hybrid_model = AcceleratedLLaMA3(
        model_path=model_path, quantize=quantize, device=device if device != "auto" else "cuda"
    )

    return InferenceEngine("hybrid", hybrid_model, hybrid_model.tokenizer, device, 0, args)
