import ast
import inspect
import json
import os
from typing import Any, Optional, get_type_hints

import requests
from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as _PydanticValidationError


class ValidationError(Exception):
    """Exception representing LLM output validation error"""

    def __init__(self, original: _PydanticValidationError):
        super().__init__(str(original))
        self.original = original


# API key configuration
_API_KEY: Optional[str] = None


def configure(api_key: str | None = None) -> None:
    """
    Configure dariko settings.

    Args:
        api_key: LLM API key. If None, attempts to get from DARIKO_API_KEY environment variable.
    """
    global _API_KEY
    _API_KEY = api_key or os.getenv("DARIKO_API_KEY")


def _get_api_key() -> str:
    """Get the configured API key. Raises error if not set."""
    if _API_KEY is None:
        raise RuntimeError(
            "API key not configured. Please set using configure() or " "set DARIKO_API_KEY environment variable."
        )
    return _API_KEY


def _infer_output_model_from_ast(frame) -> type | None:
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    try:
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename)
        # Find the last AnnAssign (typed assignment)
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


def _infer_output_model_from_locals(frame) -> type[Any] | None:
    func_name = frame.f_code.co_name
    if func_name == "<module>":
        # Use AST inference for module scope
        return _infer_output_model_from_ast(frame)
    func = frame.f_globals.get(func_name, None)
    if func is not None:
        hints = get_type_hints(func)
        if len(hints) == 1:
            return next(iter(hints.values()))
    hints = frame.f_locals.get("__annotations__", {})
    if len(hints) == 1:
        return next(iter(hints.values()))
    # fallback: use AST inference
    return _infer_output_model_from_ast(frame)


def _infer_output_model_from_return_type(frame) -> type[Any] | None:
    """
    Get the return type annotation of the caller function.
    """
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


def _get_pydantic_model(model):
    origin = getattr(model, "__origin__", None)
    if origin in (list, list):
        model = model.__args__[0]
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("output_model must be a Pydantic model")
    return model


def ask(prompt: str, *, output_model: type[Any] | None = None) -> Any:
    """
    Send prompt to LLM and return validated object using output_model.
    If output_model is not specified, infers from caller's local variable annotations.
    If called within a function, also considers return type annotation.

    Args:
        prompt: Prompt to send to LLM
        output_model: Output type. If not specified, attempts automatic inference.

    Returns:
        Object validated by output_model

    Raises:
        ValidationError: If type validation fails
        TypeError: If type annotation cannot be obtained
        RuntimeError: If API key is not configured
    """
    model = output_model
    if model is None:
        caller_frame = inspect.currentframe().f_back  # One frame up
        if caller_frame is None:
            raise TypeError("Could not get type annotation. Please specify output_model.")

        # Try local variable type annotation
        model = _infer_output_model_from_locals(caller_frame)

        # Try return type annotation
        if model is None:
            model = _infer_output_model_from_return_type(caller_frame)

    if model is None:
        raise TypeError("Could not get type annotation. Please specify output_model.")

    # Get Pydantic model
    pyd_model = _get_pydantic_model(model)

    # Get API key
    api_key = _get_api_key()

    # Call LLM API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": f"Please respond according to the following JSON schema:\n{pyd_model.model_json_schema()}",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        },
    )

    if response.status_code != 200:
        raise RuntimeError(f"LLM API call failed: {response.text}")

    try:
        llm_raw_output = response.json()["choices"][0]["message"]["content"]
        llm_raw_output = json.loads(llm_raw_output)
        llm_raw_output["api_key"] = api_key  # Add API key
        return TypeAdapter(model).validate_python(llm_raw_output)
    except _PydanticValidationError as e:
        raise ValidationError(e) from None
    except json.JSONDecodeError as e:
        raise ValidationError(
            _PydanticValidationError.from_exception_data(
                "JSONDecodeError",
                [{"loc": (), "msg": f"Could not parse LLM output as JSON: {e!s}", "type": "value_error"}],
            )
        ) from None


def ask_batch(prompts: list[str], *, output_model: type[Any] | None = None) -> list[Any]:
    """
    Execute multiple prompts in batch.

    Args:
        prompts: List of prompts to send to LLM
        output_model: Output type. If not specified, attempts automatic inference.

    Returns:
        List of objects validated by output_model

    Raises:
        ValidationError: If type validation fails
        TypeError: If type annotation cannot be obtained
        RuntimeError: If API key is not configured
    """
    # Get type annotation
    model = output_model
    if model is None:
        caller_frame = inspect.currentframe().f_back
        if caller_frame is None:
            raise TypeError("Could not get type annotation. Please specify output_model.")

        # Try local variable type annotation
        model = _infer_output_model_from_locals(caller_frame)

        # Try return type annotation
        if model is None:
            model = _infer_output_model_from_return_type(caller_frame)

    if model is None:
        raise TypeError("Could not get type annotation. Please specify output_model.")

    # Get Pydantic model
    pyd_model = _get_pydantic_model(model)

    # Get API key
    api_key = _get_api_key()

    # Call LLM API for each prompt
    results = []
    for prompt in prompts:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": f"Please respond according to the following JSON schema:\n{pyd_model.model_json_schema()}",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"LLM API call failed: {response.text}")

        try:
            llm_raw_output = response.json()["choices"][0]["message"]["content"]
            llm_raw_output = json.loads(llm_raw_output)
            llm_raw_output["api_key"] = api_key  # Add API key
            llm_raw_output["api_key"] = api_key  # APIキーを追加
            result = TypeAdapter(model).validate_python(llm_raw_output)
            results.append(result)
        except _PydanticValidationError as e:
            raise ValidationError(e) from None
        except json.JSONDecodeError as e:
            raise ValidationError(
                _PydanticValidationError.from_exception_data(
                    "JSONDecodeError",
                    [{"loc": (), "msg": f"LLMの出力がJSONとして解析できませんでした: {e!s}", "type": "value_error"}],
                )
            ) from None

    return results
