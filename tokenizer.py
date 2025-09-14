"""
Tokenizer module for LLaMA 3 models.
"""

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, model_path: str):
        mergeable_ranks = load_tiktoken_bpe(model_path)
        self.num_base_tokens = len(mergeable_ranks)
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

        self.model = tiktoken.Encoding(
            name=model_path, pat_str=self.pat_str, mergeable_ranks=mergeable_ranks, special_tokens=self.special_tokens
        )

    @property
    def bos_id(self):
        return self.special_tokens["<|begin_of_text|>"]

    @property
    def stop_tokens(self):
        return {self.special_tokens["<|end_of_text|>"], self.special_tokens["<|eot_id|>"]}

    def decode(self, toks):
        return self.model.decode([t for t in toks if t < self.num_base_tokens])

    def encode(self, text, allow_special=False):
        return self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())
