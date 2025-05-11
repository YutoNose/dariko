import inspect
import os
from typing import Any, Type, get_type_hints, Optional

from pydantic import ValidationError as _PydanticValidationError
from pydantic import TypeAdapter


class ValidationError(Exception):
    """LLM 出力の型検証エラーを表す例外"""

    def __init__(self, original: _PydanticValidationError):
        super().__init__(str(original))
        self.original = original


# APIキー設定
_API_KEY: Optional[str] = None


def configure(api_key: str | None = None) -> None:
    """
    dariko の設定を行う。
    
    Args:
        api_key: LLM APIのキー。Noneの場合は環境変数 DARIKO_API_KEY から取得を試みる。
    """
    global _API_KEY
    _API_KEY = api_key or os.getenv("DARIKO_API_KEY")


def _get_api_key() -> str:
    """設定されたAPIキーを取得する。未設定の場合はエラーを投げる。"""
    if _API_KEY is None:
        raise RuntimeError(
            "APIキーが設定されていません。configure() で設定するか、"
            "環境変数 DARIKO_API_KEY を設定してください。"
        )
    return _API_KEY


def _infer_output_model_from_locals(frame) -> Type[Any] | None:
    """
    呼び出し元フレームのローカル変数型ヒント (__annotations__)
    から戻り値を受け取る変数の型を推測する。
    最低限の実装として「注釈が 1 個だけならそれ」と割り切る。
    """
    hints: dict[str, Any] = frame.f_locals.get("__annotations__", {})
    return next(iter(hints.values())) if len(hints) == 1 else None


def _infer_output_model_from_return_type(frame) -> Type[Any] | None:
    """
    呼び出し元関数の戻り値型アノテーションを取得する。
    """
    try:
        # 呼び出し元の関数オブジェクトを取得
        caller_frame = frame.f_back
        if caller_frame is None:
            return None
        
        # 関数名を取得
        func_name = caller_frame.f_code.co_name
        if func_name == "<module>":
            return None
        
        # 関数オブジェクトを取得
        func = caller_frame.f_locals.get(func_name)
        if func is None:
            return None
        
        # 戻り値型を取得
        hints = get_type_hints(func)
        return hints.get("return")
    except Exception:
        return None


def ask(prompt: str, *, output_model: Type[Any] | None = None) -> Any:
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
        model = _infer_output_model_from_locals(caller_frame)
        
        # 戻り値型アノテーションを試す
        if model is None:
            model = _infer_output_model_from_return_type(caller_frame)

    if model is None:
        raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")

    # APIキーを取得
    api_key = _get_api_key()

    # --- ここを実際の LLM 呼び出しに差し替える ----------------------------
    # 注: 最小構成ではダミーの JSON を返す
    llm_raw_output = {"dummy": True, "prompt": prompt, "api_key": api_key}
    # ----------------------------------------------------------------------

    try:
        return TypeAdapter(model).validate_python(llm_raw_output)
    except _PydanticValidationError as e:
        raise ValidationError(e) from None


def ask_batch(prompts: list[str], *, output_model: Type[Any] | None = None) -> list[Any]:
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
        model = _infer_output_model_from_locals(caller_frame)
        
        # 戻り値型アノテーションを試す
        if model is None:
            model = _infer_output_model_from_return_type(caller_frame)

    if model is None:
        raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")

    # APIキーを取得
    api_key = _get_api_key()

    # --- ここを実際の LLM 呼び出しに差し替える ----------------------------
    # 注: 最小構成ではダミーの JSON を返す
    results = []
    for prompt in prompts:
        llm_raw_output = {"dummy": True, "prompt": prompt, "api_key": api_key}
        try:
            result = TypeAdapter(model).validate_python(llm_raw_output)
            results.append(result)
        except _PydanticValidationError as e:
            raise ValidationError(e) from None
    # ----------------------------------------------------------------------

    return results
