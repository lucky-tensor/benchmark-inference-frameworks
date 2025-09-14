#!/usr/bin/env python3
"""
Generic chat interface abstraction for different language models.

This module provides a unified interface for chat-style interactions
with different models (LLaMA 3, GPT-2, etc.) while handling model-specific
formatting requirements.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    """Standard chat message roles"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ResponseStats:
    """Statistics for a generated response"""

    response_time: float = 0.0  # Total response time in seconds
    token_count: int = 0  # Number of tokens generated
    tokens_per_second: float = 0.0  # Generation speed
    time_to_first_token: float = 0.0  # TTFT in seconds
    first_token_recorded: bool = False

    def start_generation(self):
        """Mark the start of response generation"""
        self.start_time = time.time()
        self.first_token_recorded = False

    def record_first_token(self):
        """Record when the first token was generated"""
        if not self.first_token_recorded:
            self.time_to_first_token = time.time() - self.start_time
            self.first_token_recorded = True

    def record_token(self):
        """Record that a token was generated"""
        self.token_count += 1

    def finalize(self):
        """Calculate final statistics"""
        self.response_time = time.time() - self.start_time
        if self.response_time > 0:
            self.tokens_per_second = self.token_count / self.response_time

    def format_stats(self) -> str:
        """Format statistics for display"""
        if self.token_count == 0:
            return "No tokens generated"

        return (
            f"⏱️  Response: {self.response_time:.2f}s | "
            f"🎯 Tokens: {self.token_count} | "
            f"🚀 Speed: {self.tokens_per_second:.1f} tok/s | "
            f"⚡ TTFT: {self.time_to_first_token:.2f}s"
        )


@dataclass
class ChatMessage:
    """Represents a single chat message"""

    role: MessageRole
    content: str
    stats: ResponseStats | None = None  # Statistics for generated responses

    def __post_init__(self):
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)


@dataclass
class ChatSession:
    """Represents a complete chat conversation"""

    messages: list[ChatMessage]
    system_prompt: str | None = None

    def add_message(self, role: MessageRole | str, content: str, stats: ResponseStats | None = None):
        """Add a message to the conversation"""
        self.messages.append(ChatMessage(role, content, stats))

    def get_last_message(self) -> ChatMessage | None:
        """Get the most recent message"""
        return self.messages[-1] if self.messages else None

    def get_last_response_stats(self) -> ResponseStats | None:
        """Get statistics from the last assistant response"""
        for message in reversed(self.messages):
            if message.role == MessageRole.ASSISTANT and message.stats:
                return message.stats
        return None

    def get_messages_by_role(self, role: MessageRole) -> list[ChatMessage]:
        """Get all messages from a specific role"""
        return [msg for msg in self.messages if msg.role == role]


class ChatInterface(ABC):
    """Abstract base class for model-specific chat interfaces"""

    @abstractmethod
    def encode_chat_session(self, session: ChatSession) -> list[int]:
        """
        Convert a chat session to model-specific token sequence.

        Args:
            session: The chat session to encode

        Returns:
            List of token IDs ready for model input
        """

    @abstractmethod
    def encode_message(self, message: ChatMessage) -> list[int]:
        """
        Encode a single message to tokens.

        Args:
            message: The message to encode

        Returns:
            List of token IDs for this message
        """

    @abstractmethod
    def prepare_generation_context(self, session: ChatSession) -> tuple[list[int], str]:
        """
        Prepare the context for generating the next response.

        Args:
            session: Current chat session

        Returns:
            Tuple of (token_ids, expected_role) where expected_role indicates
            what role should respond next
        """

    @abstractmethod
    def get_stop_tokens(self) -> list[int]:
        """Get tokens that should stop generation"""

    @abstractmethod
    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode tokens back to text"""

    @abstractmethod
    def is_interactive_mode_supported(self) -> bool:
        """Whether this interface supports interactive Q&A mode"""

    def format_interactive_prompt(self, prompt_text: str = "Q: ") -> str:
        """Format the interactive prompt text (override if needed)"""
        return prompt_text

    def create_response_stats(self) -> ResponseStats:
        """Create a new ResponseStats object for tracking generation"""
        return ResponseStats()


class LLaMA3ChatInterface(ChatInterface):
    """LLaMA 3 specific chat interface with structured message format"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_role(self, role: str) -> list[int]:
        """Encode a role header with LLaMA 3 special tokens"""
        return (
            [self.tokenizer.special_tokens["<|start_header_id|>"]]
            + self.tokenizer.encode(role)
            + [self.tokenizer.special_tokens["<|end_header_id|>"]]
            + self.tokenizer.encode("\n\n")
        )

    def encode_message(self, message: ChatMessage) -> list[int]:
        """Encode a single message with LLaMA 3 format"""
        role_str = message.role.value
        return (
            self.encode_role(role_str)
            + self.tokenizer.encode(message.content.strip())
            + [self.tokenizer.special_tokens["<|eot_id|>"]]
        )

    def encode_chat_session(self, session: ChatSession) -> list[int]:
        """Encode full chat session with BOS token"""
        tokens = [self.tokenizer.bos_id]

        # Add system prompt if provided
        if session.system_prompt:
            system_msg = ChatMessage(MessageRole.SYSTEM, session.system_prompt)
            tokens.extend(self.encode_message(system_msg))

        # Add all messages
        for message in session.messages:
            tokens.extend(self.encode_message(message))

        return tokens

    def prepare_generation_context(self, session: ChatSession) -> tuple[list[int], str]:
        """Prepare context for assistant response"""
        tokens = self.encode_chat_session(session)
        # Add assistant role header for generation
        tokens.extend(self.encode_role("assistant"))
        return tokens[:-1], "assistant"  # Remove last token for autoregressive generation

    def get_stop_tokens(self) -> list[int]:
        """LLaMA 3 stop tokens"""
        return [self.tokenizer.special_tokens["<|end_of_text|>"], self.tokenizer.special_tokens["<|eot_id|>"]]

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode tokens using LLaMA 3 tokenizer"""
        return self.tokenizer.decode(tokens)

    def is_interactive_mode_supported(self) -> bool:
        """LLaMA 3 supports interactive chat mode"""
        return True


class GPT2ChatInterface(ChatInterface):
    """GPT-2 chat interface using simple text continuation"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_message(self, message: ChatMessage) -> list[int]:
        """Encode message as simple text with role prefix"""
        if message.role == MessageRole.USER:
            text = f"Q: {message.content}\n"
        elif message.role == MessageRole.ASSISTANT:
            text = f"A: {message.content}\n"
        elif message.role == MessageRole.SYSTEM:
            text = f"System: {message.content}\n"
        else:
            text = f"{message.content}\n"

        return self.tokenizer.encode(text)

    def encode_chat_session(self, session: ChatSession) -> list[int]:
        """Encode chat session as simple text format"""
        tokens = []

        # Add system prompt if provided
        if session.system_prompt:
            system_msg = ChatMessage(MessageRole.SYSTEM, session.system_prompt)
            tokens.extend(self.encode_message(system_msg))

        # Add all messages
        for message in session.messages:
            tokens.extend(self.encode_message(message))

        return tokens

    def prepare_generation_context(self, session: ChatSession) -> tuple[list[int], str]:
        """Prepare context for assistant response"""
        tokens = self.encode_chat_session(session)
        # Add "A: " prefix for assistant response
        tokens.extend(self.tokenizer.encode("A: "))
        return tokens, "assistant"

    def get_stop_tokens(self) -> list[int]:
        """GPT-2 typically uses newlines or end-of-text as stop tokens"""
        # GPT-2 doesn't have specific stop tokens, we'll rely on detection logic
        return []

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode tokens using GPT-2 tokenizer"""
        return self.tokenizer.decode(tokens)

    def is_interactive_mode_supported(self) -> bool:
        """GPT-2 supports interactive mode with Q&A format"""
        return True

    def format_interactive_prompt(self, prompt_text: str = "Q: ") -> str:
        """GPT-2 uses Q: format for questions"""
        return prompt_text


def create_chat_interface(model_type: str, tokenizer) -> ChatInterface:
    """
    Factory function to create appropriate chat interface for model type.

    Args:
        model_type: Type of model ("llama3", "gpt2", etc.)
        tokenizer: Model-specific tokenizer

    Returns:
        Appropriate ChatInterface implementation
    """
    model_type = model_type.lower()

    if model_type.startswith("llama3") or model_type.startswith("llama-3"):
        return LLaMA3ChatInterface(tokenizer)
    if model_type.startswith("gpt2") or model_type.startswith("gpt-2"):
        return GPT2ChatInterface(tokenizer)
    raise ValueError(f"Unsupported model type: {model_type}")


# Convenience functions for common use cases
def create_simple_session(user_message: str, system_prompt: str = None) -> ChatSession:
    """Create a simple chat session with one user message"""
    session = ChatSession([], system_prompt)
    session.add_message(MessageRole.USER, user_message)
    return session


def create_qa_session(question: str, system_prompt: str = "You are a helpful assistant.") -> ChatSession:
    """Create a Q&A session with a question"""
    return create_simple_session(question, system_prompt)


def stream_generate_tokens(
    model_fn,
    initial_context,
    chat_interface: ChatInterface,
    stats: ResponseStats,
    max_tokens: int = 100,
    stop_on_newline: bool = False,
) -> str:
    """
    Common token-by-token generation function for both LLaMA 3 and GPT-2.

    Args:
        model_fn: Function that takes (tokens, start_pos, temperature) and returns next token
        initial_context: Initial token sequence
        chat_interface: Chat interface for decoding and stop tokens
        stats: ResponseStats object for tracking timing
        max_tokens: Maximum tokens to generate
        stop_on_newline: Stop generation on newline for GPT-2 Q&A format

    Returns:
        Generated text response
    """

    last_tok = initial_context[-1]
    start_pos = len(initial_context) - 1
    response_text = ""
    first_token = True
    token_count = 0

    while token_count < max_tokens:
        try:
            # Generate next token
            tok = model_fn(last_tok, start_pos)

            if first_token:
                stats.record_first_token()
                first_token = False
            stats.record_token()

            # Check stop conditions
            if tok in chat_interface.get_stop_tokens():
                break

            # Decode token and add to response
            decoded = chat_interface.decode_tokens([tok])
            response_text += decoded
            print(decoded, end="", flush=True)

            # GPT-2 specific: stop on newline for Q&A format
            if stop_on_newline and decoded.strip() and ("\n" in decoded):
                # Check if we've hit a Q: or A: pattern
                if response_text.strip().endswith(("Q:", "A:", "System:")):
                    break

            start_pos += 1
            last_tok = tok
            token_count += 1

        except Exception as e:
            print(f"\nGeneration stopped due to error: {e}")
            break

    return response_text.strip()
