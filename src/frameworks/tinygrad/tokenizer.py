"""TinyGrad Tokenizer Implementation

TinyGrad-specific tokenizer that integrates with the existing tokenizer
implementation while providing the standard interface.
"""

from pathlib import Path

from common.model_interface import BaseTokenizer
from common.tokenizer import Tokenizer as TinyGradTokenizerImpl


class TinyGradTokenizer(BaseTokenizer):
    """TinyGrad tokenizer wrapper providing standard interface."""

    def __init__(self, tokenizer_path: Path):
        """Initialize TinyGrad tokenizer.

        Args:
            tokenizer_path: Path to tokenizer.model file
        """
        self._tokenizer = TinyGradTokenizerImpl(str(tokenizer_path))

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs
        """
        return self._tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text string
        """
        return self._tokenizer.decode(token_ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Size of the vocabulary
        """
        return self._tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token ID."""
        return self._tokenizer.eos_id

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        # TinyGrad tokenizer doesn't have explicit pad token, use eos
        return self._tokenizer.eos_id
