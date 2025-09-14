#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys

# Add current directory to path to find extra module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Note: Models will be downloaded to ~/models/<model_name>/

with contextlib.suppress(ImportError):
    import tiktoken
from tinygrad import Device, GlobalCounters, Tensor, TinyJit, Variable, dtypes
from tinygrad.helpers import DEBUG, JIT, Timing, colored, fetch, getenv, trange
from tinygrad.nn import Embedding, LayerNorm, Linear
from tinygrad.nn.state import get_state_dict, gguf_load, load_state_dict, torch_load
from tinygrad.uop.ops import UOp

from ..common.chat_interface import (
    ChatSession,
    GPT2ChatInterface,
    MessageRole,
    create_simple_session,
)
from ..llama.extra.bench_log import BenchEvent, WallTimeEvent

MAX_CONTEXT = getenv("MAX_CONTEXT", 1024)
HALF = getenv("HALF")


class Attention:
    def __init__(self, dim, n_heads):
        self.c_attn = Linear(dim, 3 * dim, bias=True)
        self.c_proj = Linear(dim, dim, bias=True)
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

    def __call__(self, x: Tensor, start_pos: Variable, mask: Tensor | None) -> Tensor:
        if mask is not None or start_pos.val == 0:
            # no symbolic shape qkv when consuming prompts
            start_pos = start_pos.val

        if HALF:
            x = x.half()
        xqkv = self.c_attn(x)
        xq, xk, xv = [
            xqkv.shrink((None, None, (i * self.dim, (i + 1) * self.dim))).reshape(
                None, None, self.n_heads, self.head_dim
            )
            for i in range(3)
        ]
        bsz, seqlen, _, _ = xq.shape

        # create kv cache
        if not hasattr(self, "cache_kv"):
            self.cache_kv = (
                Tensor.zeros(2, bsz, MAX_CONTEXT, self.n_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
            )

        # update the cache
        self.cache_kv.shrink((None, None, (start_pos, start_pos + seqlen), None, None)).assign(
            Tensor.stack(xk, xv)
        ).realize()

        if start_pos > 0:
            keys = self.cache_kv[0].shrink((None, (0, start_pos + seqlen), None, None))
            values = self.cache_kv[1].shrink((None, (0, start_pos + seqlen), None, None))
        else:
            keys = xk
            values = xv

        xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        return self.c_proj(
            xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2).reshape(bsz, seqlen, self.dim)
        )


class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.c_fc = Linear(dim, hidden_dim, bias=True)
        self.c_proj = Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())


class TransformerBlock:
    def __init__(self, dim, n_heads, norm_eps):
        self.attn = Attention(dim, n_heads)
        self.mlp = FeedForward(dim, 4 * dim)
        self.ln_1 = LayerNorm(dim, norm_eps)
        self.ln_2 = LayerNorm(dim, norm_eps)

    def __call__(self, x: Tensor, start_pos: Variable, mask: Tensor | None):
        h = x + self.attn(self.ln_1(x), start_pos, mask).float()
        return h + self.mlp(self.ln_2(h))


class Transformer:
    def __init__(self, dim, n_heads, n_layers, norm_eps, vocab_size, max_seq_len=1024):
        self.vocab_size = vocab_size
        self.wte = Embedding(vocab_size, dim)
        self.wpe = Embedding(max_seq_len, dim)
        self.h = [TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim, norm_eps)
        self.lm_head = Linear(dim, vocab_size, bias=False)
        self.forward_jit = TinyJit(self.forward)

    def forward(self, tokens: Tensor | UOp, start_pos: Variable, temperature: float = 0.0):
        if not hasattr(self, "allpos"):
            self.allpos = Tensor.arange(0, MAX_CONTEXT).reshape(1, -1).realize()
        if isinstance(tokens, UOp):
            seqlen = 1
            tok_emb = self.wte.weight.shrink(((tokens, tokens + 1), None))
        else:
            seqlen = tokens.shape[1]
            tok_emb = self.wte(tokens)

        # not symbolic when consuming the prompt
        selected_pos = (0, seqlen) if start_pos.val == 0 else (start_pos, start_pos + 1)
        pos_emb = self.wpe(self.allpos.shrink((None, selected_pos)))

        h = tok_emb + pos_emb

        if HALF:
            h = h.half()

        mask = (
            Tensor.full((1, 1, seqlen, start_pos.val + seqlen), float("-inf"), dtype=h.dtype).triu(start_pos.val + 1)
            if seqlen > 1
            else None
        )

        for hi in self.h:
            h = hi(h, start_pos, mask)

        logits = self.lm_head(self.ln_f(h))

        if logits.shape[1] == 0:
            # special case for empty prompt
            logits = Tensor.ones((logits.shape[0], self.vocab_size), dtype=logits.dtype, device=logits.device)
        else:
            logits = logits[:, -1, :]

        ret = logits.argmax(-1) if temperature < 1e-06 else (logits / temperature).softmax().multinomial()
        return ret.flatten().realize()

    def __call__(self, tokens: Tensor | UOp, start_pos: Variable, temperature: float = 0.0) -> Tensor:
        forward = self.forward_jit if JIT and (isinstance(tokens, UOp) or tokens.shape[1] == 1) else self.forward
        return forward(tokens, start_pos, temperature)


VOCAB_SIZE = 50257
MODEL_PARAMS = {
    "gpt2": {"n_layers": 12, "n_heads": 12, "dim": 768, "norm_eps": 1e-5, "vocab_size": VOCAB_SIZE},  # 124M params
    "gpt2-medium": {"n_layers": 24, "n_heads": 16, "dim": 1024, "norm_eps": 1e-5, "vocab_size": VOCAB_SIZE},  # 350M params
    "gpt2-large": {"n_layers": 36, "n_heads": 20, "dim": 1280, "norm_eps": 1e-5, "vocab_size": VOCAB_SIZE},  # 774M params
    "gpt2-xl": {"n_layers": 48, "n_heads": 25, "dim": 1600, "norm_eps": 1e-5, "vocab_size": VOCAB_SIZE},  # 1558M params
}


class GPT2:
    @staticmethod
    def build(model_size="gpt2"):
        tokenizer = tiktoken.get_encoding("gpt2")

        model = Transformer(**MODEL_PARAMS[model_size])
        model_dir = os.path.expanduser(f"~/models/{model_size}")
        os.makedirs(model_dir, exist_ok=True)
        weights = torch_load(
            fetch(
                f"https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin",
                name=os.path.join(model_dir, "pytorch_model.bin"),
            )
        )
        # special treatment for the Conv1D weights we need to transpose
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T
        # lm head and wte are tied
        weights["lm_head.weight"] = weights["wte.weight"]

        with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
            load_state_dict(model, weights)

            if HALF:
                for l in get_state_dict(model).values():
                    l.replace(l.half().realize())

        return GPT2(model, tokenizer)

    @staticmethod
    def build_gguf(model_size: str):
        q_type = model_size[len("gpt2_gguf_") :].upper()
        model_dir = os.path.expanduser(f"~/models/gpt2-{q_type.lower()}")
        os.makedirs(model_dir, exist_ok=True)
        fn = fetch(
            f"https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.{q_type}.gguf?download=true",
            name=os.path.join(model_dir, f"gpt2.{q_type}.gguf"),
        )
        gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
        kv_data, state_dict = gguf_load(gguf_tensor)

        gpt2_params = {
            "dim": kv_data["gpt2.embedding_length"],
            "n_heads": kv_data["gpt2.attention.head_count"],
            "n_layers": kv_data["gpt2.block_count"],
            "norm_eps": kv_data["gpt2.attention.layer_norm_epsilon"],
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": kv_data["gpt2.context_length"],
        }

        def _remap_gguf_key(key: str):
            replaces = [
                ("blk.", "h."),
                (".attn_qkv.bias", ".attn.c_attn.bias"),
                (".attn_qkv.weight", ".attn.c_attn.weight"),
                (".ffn_norm.bias", ".ln_2.bias"),
                (".ffn_norm.weight", ".ln_2.weight"),
                (".attn_norm.bias", ".ln_1.bias"),
                (".attn_norm.weight", ".ln_1.weight"),
                (".attn_output.bias", ".attn.c_proj.bias"),
                (".attn_output.weight", ".attn.c_proj.weight"),
                (".ffn_up.bias", ".mlp.c_fc.bias"),
                (".ffn_up.weight", ".mlp.c_fc.weight"),
                (".ffn_down.bias", ".mlp.c_proj.bias"),
                (".ffn_down.weight", ".mlp.c_proj.weight"),
                ("token_embd.weight", "wte.weight"),
                ("output.weight", "lm_head.weight"),
                ("output_norm.bias", "ln_f.bias"),
                ("output_norm.weight", "ln_f.weight"),
                ("position_embd.weight", "wpe.weight"),
            ]
            for ostr, ns in replaces:
                key = key.replace(ostr, ns)
            return key

        state_dict = {_remap_gguf_key(k): v for k, v in state_dict.items()}
        model = Transformer(**gpt2_params)
        with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
            load_state_dict(model, state_dict)
        return GPT2(model, tiktoken.get_encoding("gpt2"))

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_length: int, temperature: float, timing: bool = False, batch_size: int = 1):
        step_times = []
        prompt_tokens = self.tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        toks = [prompt_tokens[:] for _ in range(batch_size)]
        start_pos = 0
        for _ in trange(max_length, disable=(timing)):
            GlobalCounters.reset()
            if timing:
                print("")
            st = GlobalCounters.time_sum_s
            with (
                Timing(
                    "ran model in ",
                    on_exit=(
                        lambda et, start_time=st: (
                            f", {(GlobalCounters.time_sum_s - start_time) * 1e3:.2f} ms on {Device.DEFAULT}"
                            if DEBUG >= 2
                            else ""
                        )
                        + f", {GlobalCounters.global_ops * 1e-9:.2f} GOPS, {GlobalCounters.global_mem * 1e-9:.2f} GB"
                        + (
                            f", {GlobalCounters.global_mem * 1e-9 / (GlobalCounters.time_sum_s - start_time):.2f} GB/s"
                            if DEBUG >= 2
                            else ""
                        )
                    )
                    if DEBUG
                    else None,
                    enabled=timing,
                ),
                WallTimeEvent(BenchEvent.STEP),
            ):
                if batch_size == 1 and len(toks[0][start_pos:]) == 1:
                    tokens = Variable("tokens", 0, VOCAB_SIZE - 1).bind(toks[0][start_pos])
                else:
                    tokens = Tensor([x[start_pos:] for x in toks])
                tok = self.model(
                    tokens,
                    Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT - 1).bind(start_pos),
                    temperature,
                ).tolist()
            step_times.append((GlobalCounters.time_sum_s - st) * 1e3)
            start_pos = len(toks[0])
            for i, t in enumerate(tok):
                toks[i].append(t)

        if assert_time := getenv("ASSERT_MIN_STEP_TIME"):
            min_time = min(step_times)
            assert min_time < assert_time, (
                f"Speed regression, expected min step time of < {assert_time} ms but took: {min_time} ms"
            )
        return [self.tokenizer.decode(x) for x in toks]

    def generate_token(self, tokens_sequence, start_pos: int, temperature: float = 0.85):
        """Generate a single token for streaming generation"""
        from tinygrad import GlobalCounters, Tensor, Variable

        GlobalCounters.reset()

        # GPT-2 needs the slice from start_pos onward for proper context
        if isinstance(tokens_sequence, list):
            # Use the same pattern as the original: slice from start_pos
            if len(tokens_sequence[start_pos:]) == 1:
                tokens = Variable("tokens", 0, VOCAB_SIZE - 1).bind(tokens_sequence[start_pos])
            else:
                tokens = Tensor([tokens_sequence[start_pos:]])
        else:
            # Fallback for single token
            tokens = Variable("tokens", 0, VOCAB_SIZE - 1).bind(tokens_sequence)

        return self.model(
            tokens, Variable("start_pos", 1 if start_pos else 0, MAX_CONTEXT - 1).bind(start_pos), temperature
        ).item()

    def prefill(self, tokens, start_pos: int = 0):
        """Prefill the model with context tokens for faster generation"""
        from tinygrad import GlobalCounters, Variable
        from tqdm import tqdm

        # Process tokens in sequence to build up the KV cache
        for i, token in enumerate(tqdm(tokens, desc="Prefilling")):
            GlobalCounters.reset()
            pos = start_pos + i
            token_var = Variable("tokens", 0, VOCAB_SIZE - 1).bind(token)
            # Run the model to populate KV cache, don't need the output
            self.model(token_var, Variable("start_pos", 1 if pos else 0, MAX_CONTEXT - 1).bind(pos), 0.0).realize()

        return start_pos + len(tokens)


# **** main code ****

if __name__ == "__main__":
    print(f"using {Device.DEFAULT} backend")
    default_prompt = "What is the answer to life, the universe, and everything?"

    parser = argparse.ArgumentParser(
        description="Run GPT2 in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--prompt", type=str, default=default_prompt, help="Phrase to start with")
    parser.add_argument("--count", type=int, default=100, help="Max number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.85, help="Temperature in the softmax")
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-medium",
        help="Size of model to use [gpt2, gpt2-medium, gpt2-large, gpt2-xl]",
    )
    parser.add_argument("--timing", action="store_true", help="Print timing per token")
    parser.add_argument("--seed", type=int, help="Set the random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Set the input batch size")
    parser.add_argument("--benchmark", type=int, default=-1, help="Benchmark GPT with the given number of tokens")
    parser.add_argument("--noshow", action="store_true", help="Don't show the output")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode with Q&A format")
    args = parser.parse_args()

    if args.seed is not None:
        Tensor.manual_seed(args.seed)

    print(f"using {args.model_size}")
    gpt2 = GPT2.build_gguf(args.model_size) if args.model_size.startswith("gpt2_gguf_") else GPT2.build(args.model_size)

    if args.benchmark != -1:
        gpt2.model(Tensor.rand(args.batch_size, args.benchmark), Variable("a", 0, MAX_CONTEXT).bind(0)).realize()
    elif args.chat:
        # Interactive chat mode using generic chat interface
        chat_interface = GPT2ChatInterface(gpt2.tokenizer)
        session = ChatSession([], "You are a helpful assistant.")

        print("Interactive Q&A mode for GPT-2. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input(chat_interface.format_interactive_prompt())
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                # Create a simple session with the user question
                single_question_session = create_simple_session(user_input)

                # Prepare generation context
                context_tokens, expected_role = chat_interface.prepare_generation_context(single_question_session)

                # Generate response with statistics tracking
                stats = chat_interface.create_response_stats()
                stats.start_generation()

                # Convert to tensor and generate response
                context_text = chat_interface.decode_tokens(context_tokens)
                response_texts = gpt2.generate(context_text, args.count, args.temperature, timing=False, batch_size=1)

                if response_texts:
                    # Extract just the response part (after "A: ")
                    full_response = response_texts[0]
                    if full_response.startswith(context_text):
                        response_text = full_response[len(context_text) :]
                    else:
                        response_text = full_response

                    # Count tokens in response and calculate statistics
                    response_tokens = gpt2.tokenizer.encode(response_text)
                    stats.token_count = len(response_tokens)
                    stats.record_first_token()  # Simulate first token timing
                    stats.finalize()

                    # Display response
                    print(response_text.strip())
                    print(f"{stats.format_stats()}\n")

                    # Add to session for context
                    session.add_message(MessageRole.USER, user_input)
                    session.add_message(MessageRole.ASSISTANT, response_text.strip(), stats)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
    else:
        texts = gpt2.generate(args.prompt, args.count, args.temperature, timing=args.timing, batch_size=args.batch_size)
        if not args.noshow:
            print("Generating text...")
            if len(texts) == 1:
                print(texts[0])
            else:
                for i, text in enumerate(texts):
                    print(colored(f"Response {i}:", "green"), text)

        # validate output!
        if args.temperature == 0 and args.model_size == "gpt2-medium" and args.count == 10:
            expected = {
                default_prompt: "What is the answer to life, the universe, and everything?\n\n"
                "The answer is that we are all one",
                "Hello.": "Hello. I'm a little late to the party, but",
            }
            try:
                assert texts[0] == expected[args.prompt]
                print(colored("output validated", "green"))
            except KeyError:
                pass
