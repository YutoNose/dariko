# llm.py
from __future__ import annotations

import inspect
import json
from typing import Any, List, Type, Dict

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as _PydanticValidationError

from .config import get_llm_key, get_model
from .exceptions import ValidationError
from .model_utils import get_pydantic_model, infer_output_model
from .models.llm import LLM
from .models.gpt import GPT
from .models.gemma import Gemma
from .models.claude import Claude

# モデル名とLLMクラスのマッピング
MODEL_MAPPING: Dict[str, Type[LLM]] = {
    "gpt": GPT,
    "gemma": Gemma,
    "claude": Claude,
}

# ─────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────


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


def _get_llm_instance() -> LLM:
    """
    設定に基づいて適切なLLMインスタンスを返す
    """
    model_name = get_model()
    llm_key = get_llm_key()

    # モデル名からLLMクラスを特定
    for prefix, llm_class in MODEL_MAPPING.items():
        if prefix in model_name.lower():
            return llm_class.configure(model_name=model_name, llm_key=llm_key)

    raise ValueError(f"Unsupported model: {model_name}")


def _post_to_llm(messages: list[dict[str, str]]) -> str:
    """
    LLMを呼び出して content 文字列を返す。
    """
    llm = _get_llm_instance()
    return llm.call(messages)


def _parse_and_validate(raw_json: str, pyd_model: Type[BaseModel], *, llm_key: str) -> BaseModel:
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
    llm_key = get_llm_key()

    raw = _post_to_llm(
        [
            {"role": "system", "content": f"{pyd_model.model_json_schema()}"},
            {"role": "user", "content": prompt},
        ]
    )
    return _parse_and_validate(raw, pyd_model, llm_key=llm_key)


def ask_batch(prompts: List[str], *, output_model: Type[Any] | None = None) -> List[Any]:
    """
    複数プロンプトをバッチ処理し、検証済みオブジェクトをリストで返す。
    """
    pyd_model = _resolve_model(output_model)
    llm_key = get_llm_key()

    results: list[Any] = []
    for p in prompts:
        raw = _post_to_llm(
            [
                {"role": "system", "content": f"{pyd_model.model_json_schema()}"},
                {"role": "user", "content": p},
            ]
        )
        results.append(_parse_and_validate(raw, pyd_model, llm_key=llm_key))
    return results
