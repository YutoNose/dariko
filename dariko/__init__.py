from importlib.metadata import version as _v

from .config import configure
from .exceptions import ValidationError
from .llm import ask, ask_batch

__all__ = ["ask", "ask_batch", "configure", "ValidationError"]
__version__ = _v(__name__)  # pyproject.toml のバージョンを反映
