from importlib.metadata import version as _v

from .core import ask, ValidationError

__all__ = ["ask", "ValidationError"]
__version__ = _v(__name__)  # pyproject.toml のバージョンを反映
