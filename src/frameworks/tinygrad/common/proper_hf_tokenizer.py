"""
Proper HuggingFace tokenizer implementation using the actual tokenizer files.
This uses the transformers library to load the tokenizer correctly.
"""

import json
from pathlib import Path

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ProperHuggingFaceTokenizer:
    """
    Tokenizer that properly uses HuggingFace transformers library.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")

        # Load tokenizer using transformers library
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {model_dir}: {e}")

        # Load config for additional info
        config_path = self.model_dir / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Cache special tokens
        self._bos_token = self.config.get("bos_token", "<|startoftext|>")
        self._eos_token = self.config.get("eos_token", "<|endoftext|>")
        self._pad_token = self.config.get("pad_token", "<|pad|>")
        self._user_token = "<|user|>"
        self._assistant_token = "<|assistant|>"

    @property
    def bos_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        # Fallback to encoding the BOS token
        try:
            return self.tokenizer.encode(self._bos_token, add_special_tokens=False)[0]
        except:
            return 32767  # <|startoftext|> token ID from config

    @property
    def eos_id(self) -> int:
        """Return the end-of-sequence token ID."""
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        try:
            return self.tokenizer.encode(self._eos_token, add_special_tokens=False)[0]
        except:
            return 32764  # <|endoftext|> token ID from config

    @property
    def stop_tokens(self) -> set:
        """Return set of stop token IDs."""
        stop_ids = set()

        # Add EOS token
        stop_ids.add(self.eos_id)

        # Add pad token if different
        try:
            pad_id = self.tokenizer.encode(self._pad_token, add_special_tokens=False)[0]
            stop_ids.add(pad_id)
        except:
            stop_ids.add(32763)  # <|pad|> token ID from config

        return stop_ids

    @property
    def special_tokens(self) -> dict[str, int]:
        """Return mapping of special tokens to their IDs."""
        special_tokens = {}

        # Add known special tokens
        for _token_name, token_str in [
            ("bos_token", self._bos_token),
            ("eos_token", self._eos_token),
            ("pad_token", self._pad_token),
            ("user_token", self._user_token),
            ("assistant_token", self._assistant_token),
        ]:
            try:
                token_id = self.tokenizer.encode(token_str, add_special_tokens=False)[0]
                special_tokens[token_str] = token_id
            except:
                pass

        return special_tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Apply chat template if available."""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False
                )
            except:
                pass

        # Fallback to manual template
        formatted_text = f"{self._bos_token}"
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                formatted_text += f"{self._user_token}{content}"
            elif role == "assistant":
                formatted_text += f"{self._assistant_token}{content}"

        if add_generation_prompt:
            formatted_text += f"{self._assistant_token}"

        return formatted_text

    def encode_chat(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> list[int]:
        """Encode a chat conversation to token IDs."""
        formatted_text = self.apply_chat_template(messages, add_generation_prompt)
        return self.encode(formatted_text, add_special_tokens=False)
