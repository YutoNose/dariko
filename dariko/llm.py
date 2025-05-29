# llm.py
from __future__ import annotations

import inspect
import json
from typing import Any, List, Literal, Type

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as _PydanticValidationError

from .config import get_api_key
from .exceptions import ValidationError
from .model_utils import get_pydantic_model, infer_output_model
from .models.post_to_openai import post_to_openai
from .models.post_to_gemini import post_to_gemini

# ─────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────


def _resolve_model(output_model: Type[Any] | None) -> Type[BaseModel]:
    """
    output_model が None の場合は呼び出しフレームから推論し、
    最終的に Pydantic Model 型を返す。
    """
    if output_model is None:
        current_frame = inspect.currentframe()
        if current_frame is None:
            raise TypeError(
                "フレームの取得に失敗しました。output_model を指定してください。"
            )
        caller_frame = current_frame.f_back
        model = infer_output_model(caller_frame)
        if model is None:
            raise TypeError(
                "型アノテーションが取得できませんでした。output_model を指定してください。"
            )
    else:
        model = output_model
    return get_pydantic_model(model)  # 型チェックも兼ねる


def _post_to_llm(messages: list[dict[str, str]], model_api_import_name: str) -> str:
    """
    LLM APIを呼び出して content 文字列を返す。
    """
    if model_api_import_name == "openai":
        return post_to_openai(messages)
    elif model_api_import_name == "gemini":
        return post_to_gemini(messages)
    else:
        raise RuntimeError(
            f"モデルのapi呼び出しもとがなにかを確認してください。: {model_api_import_name}"
        )


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
        # デバッグのために実際のレスポンス内容を表示
        print(f"デバッグ: LLMからの生レスポンス: {raw_json}")
        print(f"デバッグ: JSONDecodeError: {e}")

        # JSONDecodeError をそのまま PydanticValidationError として扱う
        try:
            # 無効なデータで Pydantic のバリデーションを実行してエラーを発生させる
            TypeAdapter(pyd_model).validate_python({"invalid": "data"})
        except _PydanticValidationError as pyd_error:
            raise ValidationError(pyd_error) from e
        # この行は到達しないが、リンターエラーを回避するため
        # JSONDecodeErrorを RuntimeError として処理
        raise RuntimeError(f"LLMの出力がJSONとして解析できませんでした: {e}") from e
    except _PydanticValidationError as e:
        raise ValidationError(e) from None


# ─────────────────────────────────────────────────────────────
# パブリック API
# ─────────────────────────────────────────────────────────────
def ask(
    prompt: str,
    *,
    model_api_import_name: Literal["openai", "gemini"],
    output_model: Type[Any] | None = None,
) -> Any:
    """
    単一プロンプトを実行し、Pydantic 検証済みオブジェクトを返す。
    """
    pyd_model = _resolve_model(output_model)
    api_key = get_api_key()

    raw = _post_to_llm(
        [
            {"role": "system", "content": f"{pyd_model.model_json_schema()}"},
            {"role": "user", "content": prompt},
        ],
        model_api_import_name,
    )
    return _parse_and_validate(raw, pyd_model, api_key=api_key)


def ask_batch(
    prompts: List[str],
    *,
    model_api_import_name: Literal["openai", "gemini"],
    output_model: Type[Any] | None = None,
) -> List[Any]:
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
            ],
            model_api_import_name,
        )
        results.append(_parse_and_validate(raw, pyd_model, api_key=api_key))
    return results
