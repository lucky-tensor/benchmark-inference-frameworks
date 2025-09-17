"""
HuggingFace BPE tokenizer implementation for models that use vocab.json + merges.txt format.
This is specifically designed to work with models like Nikity/lille-130m-instruct.
"""

import json
from pathlib import Path


class HuggingFaceBPETokenizer:
    """
    Simple BPE tokenizer implementation compatible with HuggingFace format.
    Supports vocab.json + merges.txt files.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

        # Load vocabulary
        vocab_path = self.model_dir / "vocab.json"
        with open(vocab_path, encoding='utf-8') as f:
            self.vocab = json.load(f)

        # Create reverse mapping
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}

        # Load merges
        merges_path = self.model_dir / "merges.txt"
        self.bpe_merges = {}
        with open(merges_path, encoding='utf-8') as f:
            # Skip header line if present
            lines = f.read().strip().split('\n')
            if lines[0].startswith('#'):
                lines = lines[1:]

            for i, line in enumerate(lines):
                if line.strip():
                    pair = tuple(line.split())
                    if len(pair) == 2:
                        self.bpe_merges[pair] = i

        # Load tokenizer config for special tokens
        config_path = self.model_dir / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Set up special tokens (detect from vocab)
        # Check what special tokens are actually available
        if "<|startoftext|>" in self.vocab:
            self._bos_token = "<|startoftext|>"
        else:
            self._bos_token = self.config.get("bos_token", "<|begin_of_text|>")

        if "<|endoftext|>" in self.vocab:
            self._eos_token = "<|endoftext|>"
        else:
            self._eos_token = self.config.get("eos_token", "<|end_of_text|>")

        if "<|pad|>" in self.vocab:
            self._pad_token = "<|pad|>"
        else:
            self._pad_token = self.config.get("pad_token", "<|pad|>")

        self._unk_token = self.config.get("unk_token", "<|unk|>")

        # Create a simple word-based tokenizer for basic functionality
        # This is a simplified implementation - for production use, you'd want proper BPE

        # Add special_tokens attribute for compatibility
        self.special_tokens = {
            self._bos_token: self.bos_id,
            self._eos_token: self.vocab.get(self._eos_token, 1)
        }

    @property
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        # Try to find BOS token in vocab, fallback to token 0
        if self._bos_token in self.vocab:
            return self.vocab[self._bos_token]
        # Common BOS tokens
        for token in ["<|begin_of_text|>", "<s>", "<bos>", "[BOS]"]:
            if token in self.vocab:
                return self.vocab[token]
        return 0  # Fallback

    @property
    def stop_tokens(self) -> set:
        """Return set of stop token IDs."""
        stop_ids = set()

        # Try to find EOS token
        if self._eos_token in self.vocab:
            stop_ids.add(self.vocab[self._eos_token])

        # Common EOS tokens
        for token in ["<|end_of_text|>", "</s>", "<eos>", "[EOS]", "<|eot_id|>"]:
            if token in self.vocab:
                stop_ids.add(self.vocab[token])

        return stop_ids if stop_ids else {1}  # Fallback to token 1

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """
        Simple encoding implementation.
        For a full BPE implementation, you'd need proper BPE merge operations.
        This is a simplified version for basic functionality.
        """
        # This is a very simplified tokenization - just split by spaces and encode words
        words = text.split()
        tokens = []

        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Try to encode character by character as fallback
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        # Use unknown token
                        if self._unk_token in self.vocab:
                            tokens.append(self.vocab[self._unk_token])
                        else:
                            tokens.append(0)  # Fallback unknown token

        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.vocab_reverse:
                token = self.vocab_reverse[token_id]
                # Skip special tokens in basic decoding
                if not (token.startswith('<') and token.endswith('>')):
                    tokens.append(token)

        return ''.join(tokens).replace('Ġ', ' ').strip()  # Handle GPT-2 style space encoding
