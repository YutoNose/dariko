"""dariko: LLM の出力を Pydantic モデルで型安全に扱うライブラリ。"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from dariko.config import get_config, set_config
from dariko.driver import (
    StreamedResponse,
    ValidationError,
    aask,
    aask_batch,
    ask,
    ask_batch,
    ask_stream,
)

try:
    __version__ = _pkg_version("dariko")
except PackageNotFoundError:  # pragma: no cover - 未インストール (ソース直実行) 時
    __version__ = "0.0.0.dev0"

__all__ = [
    "StreamedResponse",
    "ValidationError",
    "__version__",
    "aask",
    "aask_batch",
    "ask",
    "ask_batch",
    "ask_stream",
    "get_config",
    "set_config",
]
