"""
Text generation utilities and functions.
"""

from tinygrad import GlobalCounters, Tensor
from tinygrad.helpers import tqdm

TEMPERATURE = 0.95
TOP_K = 0
TOP_P = 0.0
ALPHA_F = 0.0
ALPHA_P = 0.0

last_seen_toks = []


def encode_role(role: str, tokenizer):
    """Encode a role header with special tokens"""
    return (
        [tokenizer.special_tokens["<|start_header_id|>"], *tokenizer.encode(role), tokenizer.special_tokens["<|end_header_id|>"], *tokenizer.encode("\n\n")]
    )


def encode_message(role: str, content: str, tokenizer):
    """Encode a message with role and content"""
    return encode_role(role, tokenizer) + tokenizer.encode(content.strip()) + [tokenizer.special_tokens["<|eot_id|>"]]


def prefill(model, toks, start_pos=0, device_param=None):
    """Prefill model with tokens"""
    global last_seen_toks
    from tinygrad import Device

    model_device = device_param if device_param is not None else Device.DEFAULT

    if start_pos == 0:
        for i, (a, b) in enumerate(zip(toks, last_seen_toks, strict=False)):
            if a != b:
                break
        else:
            i = min(len(toks), len(last_seen_toks))
        start_pos += i
        last_seen_toks = toks
        toks = toks[i:]

    for tok in tqdm(toks):
        GlobalCounters.reset()
        model(Tensor([[tok]], device=model_device), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P).realize()
        start_pos += 1
    return start_pos
