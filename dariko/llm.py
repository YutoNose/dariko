import inspect
import json
from typing import Any

import requests
from pydantic import TypeAdapter
from pydantic import ValidationError as _PydanticValidationError

from .config import get_api_key, get_model
from .exceptions import ValidationError
from .model_utils import (
    get_pydantic_model,
    infer_output_model_from_locals,
    infer_output_model_from_return_type,
)


def ask(prompt: str, *, output_model: type[Any] | None = None) -> Any:
    """
    LLM へ prompt を投げ、output_model で検証済みのオブジェクトを返す。
    output_model が未指定なら呼び出し元のローカル変数アノテーションを推測。
    関数内で呼ばれた場合は戻り値型アノテーションも考慮する。

    Args:
        prompt: LLMに送信するプロンプト
        output_model: 出力の型。未指定の場合は自動推論を試みる。

    Returns:
        output_model で検証済みのオブジェクト

    Raises:
        ValidationError: 型検証に失敗した場合
        TypeError: 型アノテーションが取得できなかった場合
        RuntimeError: APIキーが設定されていない場合
    """
    model = output_model
    if model is None:
        caller_frame = inspect.currentframe().f_back  # 1 つ上のフレーム
        if caller_frame is None:
            raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")

        # ローカル変数の型アノテーションを試す
        model = infer_output_model_from_locals(caller_frame)

        # 戻り値型アノテーションを試す
        if model is None:
            model = infer_output_model_from_return_type(caller_frame)

    if model is None:
        raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")

    # Pydanticモデルを取得
    get_pydantic_model(model)

    # APIキーを取得
    api_key = get_api_key()

    # LLM APIを呼び出す
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": get_model(),
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        },
    )

    if response.status_code != 200:
        raise RuntimeError(f"LLM API呼び出しに失敗しました: {response.text}")

    try:
        llm_raw_output = response.json()["choices"][0]["message"]["content"]
        llm_raw_output = json.loads(llm_raw_output)
        llm_raw_output["api_key"] = api_key  # APIキーを追加
        return TypeAdapter(model).validate_python(llm_raw_output)
    except _PydanticValidationError as e:
        raise ValidationError(e) from None
    except json.JSONDecodeError as e:
        raise ValidationError(
            _PydanticValidationError.from_exception_data(
                "JSONDecodeError",
                [{"loc": (), "msg": f"LLMの出力がJSONとして解析できませんでした: {e!s}", "type": "value_error"}],
            )
        ) from None


def ask_batch(prompts: list[str], *, output_model: type[Any] | None = None) -> list[Any]:
    """
    複数のプロンプトをバッチ処理で実行する。

    Args:
        prompts: LLMに送信するプロンプトのリスト
        output_model: 出力の型。未指定の場合は自動推論を試みる。

    Returns:
        output_model で検証済みのオブジェクトのリスト

    Raises:
        ValidationError: 型検証に失敗した場合
        TypeError: 型アノテーションが取得できなかった場合
        RuntimeError: APIキーが設定されていない場合
    """
    # 型アノテーションの取得
    model = output_model
    if model is None:
        caller_frame = inspect.currentframe().f_back
        if caller_frame is None:
            raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")

        # ローカル変数の型アノテーションを試す
        model = infer_output_model_from_locals(caller_frame)

        # 戻り値型アノテーションを試す
        if model is None:
            model = infer_output_model_from_return_type(caller_frame)

    if model is None:
        raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")

    # Pydanticモデルを取得
    pyd_model = get_pydantic_model(model)

    # APIキーを取得
    api_key = get_api_key()

    # 各プロンプトに対してLLM APIを呼び出す
    results = []
    for prompt in prompts:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": get_model(),
                "messages": [
                    {
                        "role": "system",
                        "content": f"{pyd_model.model_json_schema()}",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"LLM API呼び出しに失敗しました: {response.text}")

        try:
            llm_raw_output = response.json()["choices"][0]["message"]["content"]
            llm_raw_output = json.loads(llm_raw_output)
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
