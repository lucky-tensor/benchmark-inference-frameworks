#!/usr/bin/env python3
"""
LLaMA 3 implementation using tinygrad.

This is the main entry point for running LLaMA 3 models with various configurations.
"""

import argparse
import os
import sys
from pathlib import Path

# Set custom download directory
os.environ["TINYGRAD_DOWNLOAD_CACHE"] = os.path.expanduser("~/models")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tinygrad import Device, Tensor
from tinygrad.nn.state import get_parameters

from chat_interface import (
    ChatMessage,
    ChatSession,
    LLaMA3ChatInterface,
    MessageRole,
)
from model_config import build_transformer, resolve_model_path
from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_model", action="store_true", help="Download a model")
    parser.add_argument("--model", type=Path, help="Model path")
    parser.add_argument("--size", choices=["1B", "8B", "70B", "405B"], default="1B", help="Model size")
    parser.add_argument("--shard", type=int, default=1, help="Shard the model across multiple devices")
    parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
    parser.add_argument(
        "--no_api", action="store_true", default=True, help="Disable the api and run a cli test interface"
    )
    parser.add_argument("--api", action="store_false", dest="no_api", help="Enable the web API (requires bottle)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web server bind address")
    parser.add_argument("--port", type=int, default=7776, help="Web server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature")
    parser.add_argument("--benchmark", action="store_true", help="Run a benchmark")
    parser.add_argument("--timing", action="store_true", help="Print timing per token")
    parser.add_argument("--profile", action="store_true", help="Output profile data")
    args = parser.parse_args()

    # Resolve model path
    args.model = resolve_model_path(args.model, args.size, args.download_model)
    assert args.model is not None, "please provide --model option"

    # Set random seed
    if args.seed is not None:
        Tensor.manual_seed(args.seed)
    if args.benchmark:
        Tensor.manual_seed(42)
    print(f"seed = {Tensor._seed}")

    # Update global temperature
    import generation

    generation.TEMPERATURE = args.temperature

    # Initialize tokenizer and model
    tokenizer = Tokenizer(str((args.model if args.model.is_dir() else args.model.parent) / "tokenizer.model"))
    device = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard)) if args.shard > 1 else Device.DEFAULT
    model = build_transformer(args.model, model_size=args.size, quantize=args.quantize, device=device)
    param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(model))

    # Route to appropriate mode
    if not args.no_api and not args.benchmark:
        from web_api import create_web_api

        app = create_web_api(model, tokenizer, device, args)
        app.run(host=args.host, port=args.port, debug=args.debug)
    elif args.benchmark:
        from benchmark import run_benchmark

        run_benchmark(model, tokenizer, args, param_bytes, device)
    else:
        run_interactive_mode(model, tokenizer, device, args, param_bytes)


def run_interactive_mode(model, tokenizer, device, args, param_bytes):
    """Run interactive chat mode with the model"""
    from tinygrad import GlobalCounters
    from tinygrad.helpers import DEBUG, Profiling, Timing

    from generation import prefill

    chat_interface = LLaMA3ChatInterface(tokenizer)
    session = ChatSession([], "You are an helpful assistant.")

    initial_tokens = chat_interface.encode_chat_session(session)
    start_pos = prefill(model, initial_tokens)

    print("Interactive Q&A mode. Type 'quit' to exit.\\n")

    while True:
        try:
            user_input = input(chat_interface.format_interactive_prompt())
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            session.add_message(MessageRole.USER, user_input)
            context_tokens, expected_role = chat_interface.prepare_generation_context(session)

            stats = chat_interface.create_response_stats()
            stats.start_generation()

            if len(context_tokens) > len(initial_tokens):
                new_tokens = context_tokens[len(initial_tokens) :]
                start_pos = prefill(model, new_tokens, start_pos=start_pos)
                initial_tokens = context_tokens

            last_tok = context_tokens[-1]
            response_text = ""
            first_token = True

            while True:
                GlobalCounters.reset()
                if args.timing or args.profile:
                    print("")
                st = GlobalCounters.time_sum_s

                with Profiling(enabled=args.profile):
                    with Timing(
                        "total ",
                        enabled=args.timing,
                        on_exit=lambda x: f", {1e9 / x:.2f} tok/s, {GlobalCounters.global_mem / x:.2f} GB/s, param {param_bytes / x:.2f} GB/s",
                    ):
                        with Timing(
                            "enqueue in ",
                            enabled=args.timing,
                            on_exit=(
                                lambda et: (
                                    f", {(GlobalCounters.time_sum_s - st) * 1e3:.2f} ms on {Device.DEFAULT}"
                                    if DEBUG >= 2
                                    else ""
                                )
                                + f", {GlobalCounters.global_ops * 1e-9:.2f} GOPS, {GlobalCounters.global_mem * 1e-9:.2f} GB"
                                + (
                                    f", {GlobalCounters.global_mem * 1e-9 / (GlobalCounters.time_sum_s - st):.2f} GB/s, param {param_bytes * 1e-9 / (GlobalCounters.time_sum_s - st):.2f} GB/s"
                                    if DEBUG >= 2
                                    else ""
                                )
                            )
                            if DEBUG
                            else None,
                        ):
                            import generation

                            tok = model(
                                Tensor([[last_tok]], device=device),
                                start_pos,
                                generation.TEMPERATURE,
                                generation.TOP_K,
                                generation.TOP_P,
                                generation.ALPHA_F,
                                generation.ALPHA_P,
                            )
                        tok = tok.item()

                if first_token:
                    stats.record_first_token()
                    first_token = False
                stats.record_token()

                start_pos += 1
                last_tok = tok

                if tok in chat_interface.get_stop_tokens():
                    break

                decoded = chat_interface.decode_tokens([tok])
                response_text += decoded
                print(decoded, end="", flush=True)

            stats.finalize()
            print(f"\\n{stats.format_stats()}\\n")

            session.add_message(MessageRole.ASSISTANT, response_text, stats)
            initial_tokens.extend(chat_interface.encode_message(ChatMessage(MessageRole.USER, user_input)))
            initial_tokens.extend(chat_interface.encode_message(ChatMessage(MessageRole.ASSISTANT, response_text)))

        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break
        except EOFError:
            print("\\nGoodbye!")
            break


if __name__ == "__main__":
    main()
