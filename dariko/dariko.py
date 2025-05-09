import inspect
from typing import Any, Type

from pydantic import ValidationError as _PydanticValidationError
from pydantic import TypeAdapter


class ValidationError(Exception):
    """LLM 出力の型検証エラーを表す例外"""

    def __init__(self, original: _PydanticValidationError):
        super().__init__(str(original))
        self.original = original


def _infer_output_model_from_locals(frame) -> Type[Any] | None:
    """
    呼び出し元フレームのローカル変数型ヒント (__annotations__)
    から戻り値を受け取る変数の型を推測する。
    最低限の実装として「注釈が 1 個だけならそれ」と割り切る。
    """
    hints: dict[str, Any] = frame.f_locals.get("__annotations__", {})
    return next(iter(hints.values())) if len(hints) == 1 else None


def ask(prompt: str, *, output_model: Type[Any] | None = None) -> Any:
    """
    LLM へ prompt を投げ、output_model で検証済みのオブジェクトを返す。
    output_model が未指定なら呼び出し元のローカル変数アノテーションを推測。
    """
    model = output_model
    if model is None:
        caller_frame = inspect.currentframe().f_back  # 1 つ上のフレーム
        model = _infer_output_model_from_locals(caller_frame)

    if model is None:
        raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")

    # --- ここを実際の LLM 呼び出しに差し替える ----------------------------
    # 注: 最小構成ではダミーの JSON を返す
    llm_raw_output = {"dummy": True, "prompt": prompt}
    # ----------------------------------------------------------------------

    try:
        return TypeAdapter(model).validate_python(llm_raw_output)
    except _PydanticValidationError as e:
        raise ValidationError(e) from None
