import ast
from typing import Any, get_type_hints

from pydantic import BaseModel


def infer_output_model_from_ast(frame) -> type | None:
    """Infer type annotation using AST."""
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    try:
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename)
        # Find the last AnnAssign (type annotated assignment)
        last_ann = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and hasattr(node, "lineno") and node.lineno < lineno:
                if last_ann is None or node.lineno > last_ann.lineno:
                    last_ann = node
        if last_ann and isinstance(last_ann.target, ast.Name):
            type_str = ast.unparse(last_ann.annotation)
            try:
                model = eval(type_str, frame.f_globals, frame.f_locals)
                if isinstance(model, type) and issubclass(model, BaseModel):
                    return model
                return None
            except Exception:
                return None
    except Exception:
        return None
    return None


def infer_output_model_from_locals(frame) -> type[Any] | None:
    """Infer type from local variable annotations."""
    func_name = frame.f_code.co_name
    if func_name == "<module>":
        # Use AST inference for module scope
        return infer_output_model_from_ast(frame)
    func = frame.f_globals.get(func_name, None)
    if func is not None:
        hints = get_type_hints(func)
        if len(hints) == 1:
            return next(iter(hints.values()))
    hints = frame.f_locals.get("__annotations__", {})
    if len(hints) == 1:
        return next(iter(hints.values()))
    # fallback: use AST inference
    return infer_output_model_from_ast(frame)


def infer_output_model_from_return_type(frame) -> type[Any] | None:
    """Get return type annotation from the caller function."""
    try:
        # Get caller function object
        caller_frame = frame.f_back
        if caller_frame is None:
            return None

        # Get function name
        func_name = caller_frame.f_code.co_name
        if func_name == "<module>":
            return None

        # Get function object
        func = caller_frame.f_locals.get(func_name)
        if func is None:
            return None

        # Get return type
        hints = get_type_hints(func)
        return hints.get("return")
    except Exception:
        return None


def get_pydantic_model(model):
    """Get Pydantic model."""
    origin = getattr(model, "__origin__", None)
    if origin in (list, list):
        model = model.__args__[0]
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("output_model must be a Pydantic model")
    return model
