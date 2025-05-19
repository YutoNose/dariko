# llm.py
from __future__ import annotations

import inspect
import json
from typing import Any, List, Type

import requests
from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as _PydanticValidationError

from .config import get_api_key, get_model
from .exceptions import ValidationError
from .model_utils import infer_output_model, get_pydantic_model


# ─────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _resolve_model(output_model: Type[Any] | None) -> Type[BaseModel]:
    """
    output_model が None の場合は呼び出しフレームから推論し、
    最終的に Pydantic Model 型を返す。
    """
    if output_model is None:
        caller_frame = inspect.currentframe().f_back
        model = infer_output_model(caller_frame)
        if model is None:
            raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")
    else:
        model = output_model
    return get_pydantic_model(model)  # 型チェックも兼ねる


def _post_to_llm(messages: list[dict[str, str]]) -> str:
    """
    OpenAI ChatCompletion を呼び出して content 文字列を返す。
    """
    api_key = get_api_key()
    r = requests.post(
        _OPENAI_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": get_model(),
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"LLM API呼び出しに失敗しました: {r.text}")
    return r.json()["choices"][0]["message"]["content"]


def _parse_and_validate(
    raw_json: str, pyd_model: Type[BaseModel], *, api_key: str
) -> BaseModel:
    """
    LLM 出力(JSON文字列)を parse & Pydantic 検証。
    成功すれば Pydantic モデルのインスタンスを返す。
    """
    try:
        data = json.loads(raw_json)
        return TypeAdapter(pyd_model).validate_python(data)
    except json.JSONDecodeError as e:
        raise ValidationError(
            _PydanticValidationError.from_exception_data(
                "JSONDecodeError",
                [{"loc": (), "msg": f"LLMの出力がJSONとして解析できませんでした: {e}", "type": "value_error"}],
            )
        ) from None
    except _PydanticValidationError as e:
        raise ValidationError(e) from None


# ─────────────────────────────────────────────────────────────
# パブリック API
# ─────────────────────────────────────────────────────────────
def ask(prompt: str, *, output_model: Type[Any] | None = None) -> Any:
    """
    単一プロンプトを実行し、Pydantic 検証済みオブジェクトを返す。
    """
    pyd_model = _resolve_model(output_model)
    api_key = get_api_key()

    raw = _post_to_llm(
        [
            {"role": "system", "content": f"{pyd_model.model_json_schema()}"},
            {"role": "user", "content": prompt},
        ]
    )
    return _parse_and_validate(raw, pyd_model, api_key=api_key)


def ask_batch(prompts: List[str], *, output_model: Type[Any] | None = None) -> List[Any]:
    """
    複数プロンプトをバッチ処理し、検証済みオブジェクトをリストで返す。
    """
    pyd_model = _resolve_model(output_model)
    api_key = get_api_key()

    results: list[Any] = []
    for p in prompts:
        raw = _post_to_llm(
            [
                {"role": "system", "content": f"{pyd_model.model_json_schema()}"},
                {"role": "user", "content": p},
            ]
        )
        results.append(_parse_and_validate(raw, pyd_model, api_key=api_key))
    return results
