"""LLM プロバイダ実装のエクスポート。

``Gemma`` は torch/transformers を必要とするため遅延ロードする
(``from dariko.models import Gemma`` した時点では import しない)。
"""

from typing import Any

from .claude import Claude
from .gpt import GPT
from .llm import LLM

__all__ = ["GPT", "LLM", "Claude", "Gemma"]


def __getattr__(name: str) -> Any:
    if name == "Gemma":
        from .gemma import Gemma

        return Gemma
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
