"""
Benchmarking functionality for model performance testing.
"""

from tinygrad import Device, GlobalCounters, Tensor
from tinygrad.helpers import DEBUG, Profiling, Timing, colored

from ..llama.extra.bench_log import BenchEvent, WallTimeEvent
from .generation import TEMPERATURE, encode_message, encode_role, prefill


def run_benchmark(model, tokenizer, args, param_bytes, device):
    """Run performance benchmark"""
    toks = [tokenizer.bos_id, *encode_message("user", "Hello.", tokenizer), *encode_role("assistant", tokenizer)]

    start_pos = prefill(model, toks[:-1])
    last_tok = toks[-1]
    generated = ""

    for _ in range(20):
        GlobalCounters.reset()
        st = GlobalCounters.time_sum_s
        with Profiling(enabled=args.profile):
            with Timing(
                "total ",
                on_exit=lambda x: f", {1e9 / x:.2f} tok/s, {GlobalCounters.global_mem / x:.2f} GB/s, "
                f"param {param_bytes / x:.2f} GB/s",
            ):
                with WallTimeEvent(BenchEvent.STEP):
                    with Timing(
                        "enqueue in ",
                        on_exit=(
                            lambda et, start_time=st: (
                                f", {(GlobalCounters.time_sum_s - start_time) * 1e3:.2f} ms on {Device.DEFAULT}"
                                if DEBUG >= 2
                                else ""
                            )
                            + f", {GlobalCounters.global_ops * 1e-9:.2f} GOPS, "
                            f"{GlobalCounters.global_mem * 1e-9:.2f} GB"
                            + (
                                f", {GlobalCounters.global_mem * 1e-9 / (GlobalCounters.time_sum_s - start_time):.2f} GB/s, "
                                f"param {param_bytes * 1e-9 / (GlobalCounters.time_sum_s - start_time):.2f} GB/s"
                                if DEBUG >= 2
                                else ""
                            )
                        )
                        if DEBUG
                        else None,
                    ):
                        tok = model(Tensor([[last_tok]], device=device), start_pos, TEMPERATURE, 0, 0.0, 0.0, 0.0)
                    tok = tok.item()
        start_pos += 1
        last_tok = tok
        generated += tokenizer.decode([tok])
        print(generated)

    # Validation for specific model
    if "LLaMA-3/8B-SF-DPO" in args.model.as_posix() and (TEMPERATURE == 0.85 or TEMPERATURE == 0):
        if TEMPERATURE == 0.85:
            EXPECTED_TEXT = {
                1: "Hello! How can I help you today? If you have any questions or need assistance with anything,",
                2: "Hello! How can I help you today? If you have any questions, need assistance or just want",
                3: "Hello! How can I help you today? If you have any questions or need assistance, feel free",
                4: "Hello! How can I assist you today? If you have any questions, need information, or require",
                5: "Hello! How can I assist you today? If you have any questions or need help with something",
                6: "Hello! How can I assist you today? If you have any questions, need information, or require",
            }
        else:
            EXPECTED_TEXT = dict.fromkeys(
                range(1, 7), "Hello! How can I assist you today? If you have any questions or need help with something,"
            )

        assert generated == EXPECTED_TEXT[args.shard], f"{generated=} {EXPECTED_TEXT[args.shard]}"
        print("\n" + colored("output validated", "green"))
