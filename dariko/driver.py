from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Type

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as _PydanticValidationError

from .config import get_config, get_llm_key, get_model
from .exceptions import ValidationError
from .model_utils import get_pydantic_model, infer_output_model
from .models.llm import LLM


# モデル名のプレフィックス -> 対応する LLM クラスを返す関数のマッピング。
# Gemma (torch/transformers) を import するのは実際に使う時だけにするため遅延ロードする。
def _gpt_class() -> Type[LLM]:
    from .models.gpt import GPT

    return GPT


def _claude_class() -> Type[LLM]:
    from .models.claude import Claude

    return Claude


def _gemma_class() -> Type[LLM]:
    from .models.gemma import Gemma

    return Gemma


MODEL_MAPPING: Dict[str, Any] = {
    "gpt": _gpt_class,
    "claude": _claude_class,
    "gemma": _gemma_class,
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
    設定に基づいて適切な LLM インスタンスを生成する。
    """
    cfg = get_config()
    model_name = get_model()
    llm_key = get_llm_key()

    for prefix, llm_class_factory in MODEL_MAPPING.items():
        if prefix in model_name.lower():
            return llm_class_factory().configure(
                model_name=model_name,
                llm_key=llm_key,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.timeout,
            )

    raise ValueError(f"Unsupported model: {model_name}")


def _parse_and_validate(raw_json: str, pyd_model: Type[BaseModel]) -> BaseModel:
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


def _run(pyd_model: Type[BaseModel], prompt: str) -> Any:
    """
    1 プロンプトを実行する。検証に失敗した場合はエラー内容を添えて
    最大 ``max_retries`` 回まで LLM に再生成を促す (自己修復)。
    """
    cfg = get_config()
    llm = _get_llm_instance()
    schema = pyd_model.model_json_schema()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": f"次の JSON Schema に厳密に従い、JSON のみを返してください:\n{schema}"},
        {"role": "user", "content": prompt},
    ]

    last_error: ValidationError | None = None
    for _ in range(cfg.max_retries + 1):
        raw = llm.call(messages, response_schema=schema)
        try:
            return _parse_and_validate(raw, pyd_model)
        except ValidationError as e:
            last_error = e
            # 直前の出力とエラーを会話に追加し、修正版を要求する
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"前回の出力は検証に失敗しました: {e}\n"
                        "スキーマに厳密に従い、JSON のみを返してください。"
                    ),
                },
            ]

    assert last_error is not None
    raise last_error


# ─────────────────────────────────────────────────────────────
# パブリック API
# ─────────────────────────────────────────────────────────────
def ask(prompt: str, *, output_model: Type[Any] | None = None) -> Any:
    """
    単一プロンプトを実行し、Pydantic 検証済みオブジェクトを返す。

    検証に失敗した場合は ``set_config(max_retries=...)`` の回数だけ
    LLM へ再生成を促してから ``ValidationError`` を送出する。
    """
    pyd_model = _resolve_model(output_model)
    return _run(pyd_model, prompt)


def ask_batch(prompts: List[str], *, output_model: Type[Any] | None = None) -> List[Any]:
    """
    複数プロンプトをバッチ処理し、検証済みオブジェクトをリストで返す。
    """
    pyd_model = _resolve_model(output_model)
    return [_run(pyd_model, p) for p in prompts]
