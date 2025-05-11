from importlib.metadata import version as _v

from .core import ask, ask_batch, configure, ValidationError

__all__ = ["ask", "ask_batch", "configure", "ValidationError"]
__version__ = _v(__name__)  # pyproject.toml のバージョンを反映
