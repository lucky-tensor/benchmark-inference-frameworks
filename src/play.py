#!/usr/bin/env python3
"""
Simple CLI tool for interacting with TinyGrad models.
Allows you to submit prompts and see responses from LLaMA models.
"""

import argparse
import sys
import time
from pathlib import Path

# Add the current directory to Python path for module imports
sys.path.insert(0, str(Path(__file__).parent))


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for play mode."""
    parser = argparse.ArgumentParser(
        description="Interactive Model Playground",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chat with LLaMA3-1B using TinyGrad
  python play.py --model-path ~/models/llama3-1b-instruct --prompt "What is AI?"

  # Interactive mode
  python play.py --model-path ~/models/llama3-1b-instruct --interactive

  # Generate longer responses
  python play.py --model-path ~/models/llama3-1b-instruct --prompt "Explain Python" --max-tokens 100
""",
    )

    # Model configuration
    parser.add_argument("--model-path", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--model-size", default="1B", choices=["1B", "8B", "70B"], help="Model size (default: 1B)")

    # Generation parameters
    parser.add_argument("--prompt", type=str, help="Single prompt to process")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate (default: 50)")

    # Mode selection
    parser.add_argument("--interactive", action="store_true", help="Interactive mode for multiple prompts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # TinyGrad specific options
    parser.add_argument("--quantize", choices=["int8", "nf4", "float16"], help="Quantization method")
    parser.add_argument("--shard", type=int, default=1, help="Number of device shards")

    return parser


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 50, verbose: bool = False):
    """Generate response from model given a prompt."""
    import time

    # Import TinyGrad backend functions
    from frameworks.tinygrad.backends.tinygrad_backend import run_tinygrad_inference

    sys.path.insert(0, str(Path(__file__).parent / "frameworks" / "tinygrad"))
    from common.generation import prefill

    if verbose:
        print(f"🔧 Encoding prompt: '{prompt}'")

    # Use the same simple approach as interactive mode - just encode the prompt directly
    toks = [tokenizer.bos_id, *tokenizer.encode(prompt)]

    if verbose:
        print(f"🔧 Input tokens ({len(toks)}): {toks[:10]}{'...' if len(toks) > 10 else ''}")

    # Prefill the model
    prefill_start = time.perf_counter()
    start_pos = prefill(model, toks[:-1])
    prefill_time = time.perf_counter() - prefill_start

    print(f"\n💬 **{prompt}**\n")
    print("🤖 ", end="", flush=True)

    generated_tokens = []
    current_token = toks[-1]  # Start with the assistant role token
    generation_times = []
    generation_start = time.perf_counter()
    first_token_time = None

    # Generate tokens one by one
    for i in range(max_tokens):
        try:
            # Time each token generation
            token_start = time.perf_counter()
            next_token = run_tinygrad_inference(model, current_token, start_pos + i)
            token_time = time.perf_counter() - token_start

            generated_tokens.append(next_token)
            current_token = next_token
            generation_times.append(token_time)

            # Record first token time
            if first_token_time is None:
                first_token_time = token_time

            # Decode and print the token
            try:
                # Decode all tokens so far for better context
                full_text = tokenizer.decode(generated_tokens)
                if i == 0:
                    # First token
                    print(full_text, end="", flush=True)
                else:
                    # Print just the new part
                    prev_text = tokenizer.decode(generated_tokens[:-1])
                    new_text = full_text[len(prev_text):]
                    print(new_text, end="", flush=True)
            except Exception as decode_err:
                if verbose:
                    print(f"\n🔧 Decode error: {decode_err}")
                print(f"[{next_token}]", end=" ", flush=True)

            # Check for stop tokens (basic implementation)
            if hasattr(tokenizer, 'special_tokens'):
                # Check if this is an end-of-sequence token
                if next_token in [tokenizer.special_tokens.get('<|eot_id|>', -1),
                                tokenizer.special_tokens.get('</s>', -1),
                                tokenizer.special_tokens.get('<|endoftext|>', -1)]:
                    if verbose:
                        print(f"\n🔧 Stopped at end token: {next_token}")
                    break

        except KeyboardInterrupt:
            print("\n\n⚠️  Generation interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Generation error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            break

    # Calculate and display performance statistics
    total_generation_time = time.perf_counter() - generation_start
    num_tokens = len(generated_tokens)

    if num_tokens > 0 and generation_times:
        avg_latency_ms = (sum(generation_times) / len(generation_times)) * 1000
        peak_throughput = max(1.0 / t for t in generation_times) if generation_times else 0
        avg_throughput = num_tokens / total_generation_time if total_generation_time > 0 else 0
        first_token_latency_ms = first_token_time * 1000 if first_token_time else 0

        print("\n")
        print("📊 Generation Statistics:")
        print("=" * 40)
        print(f"Generated tokens:      {num_tokens}")
        print(f"First token:           {first_token_latency_ms:.2f}ms")
        print(f"Average latency:       {avg_latency_ms:.2f}ms per token")
        print(f"Average throughput:    {avg_throughput:.1f} tokens/second")
        print(f"Peak throughput:       {peak_throughput:.1f} tokens/second")
        print(f"Prefill time:          {prefill_time * 1000:.2f}ms")
        print(f"Total generation:      {total_generation_time * 1000:.2f}ms")

        if verbose:
            print(f"Token times: {[f'{t*1000:.1f}ms' for t in generation_times[:5]]}" +
                  ("..." if len(generation_times) > 5 else ""))
    else:
        print("\n")

    return generated_tokens


def main():
    """Main entry point for play mode."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate model path
    if not args.model_path.exists():
        print(f"❌ Model path does not exist: {args.model_path}")
        sys.exit(1)

    print(f"🚀 Loading TinyGrad LLaMA {args.model_size} model from {args.model_path}")

    try:
        # Import TinyGrad backend functions
        from frameworks.tinygrad.backends.tinygrad_backend import (
            get_tinygrad_model,
            get_tinygrad_model_info,
            get_tinygrad_tokenizer,
        )

        # Load model
        print("📥 Loading model...")
        start_time = time.time()
        model = get_tinygrad_model(
            model_size=args.model_size,
            model_path=args.model_path,
            quantize=args.quantize,
            shard=args.shard
        )
        model_load_time = time.time() - start_time
        print(f"✅ Model loaded in {model_load_time:.2f}s")

        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer_path = args.model_path / "tokenizer.model"
        if not tokenizer_path.exists():
            print(f"❌ Tokenizer not found at: {tokenizer_path}")
            sys.exit(1)

        tokenizer = get_tinygrad_tokenizer(tokenizer_path)
        print("✅ Tokenizer loaded")

        if args.verbose:
            # Print model info
            try:
                model_info = get_tinygrad_model_info(model)
                print(f"🔧 Model: {model_info['total_parameters']:,} parameters")
                print(f"🔧 Memory: {model_info['model_memory_gb']:.2f} GB")
            except Exception as e:
                if args.verbose:
                    print(f"⚠️  Could not get model info: {e}")

        # Interactive or single prompt mode
        if args.interactive:
            print("\n🎮 Interactive mode - type 'quit', 'exit', or Ctrl+C to stop")
            print("=" * 60)

            while True:
                try:
                    prompt = input("\n💭 Your prompt: ").strip()
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break

                    if prompt:
                        generate_response(
                            model, tokenizer, prompt,
                            max_tokens=args.max_tokens, verbose=args.verbose
                        )

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break

        elif args.prompt:
            # Single prompt mode
            generate_response(
                model, tokenizer, args.prompt,
                max_tokens=args.max_tokens, verbose=args.verbose
            )

        else:
            print("❌ Please provide either --prompt for single use or --interactive for interactive mode")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
