import os
from typing import Optional

# APIキー設定
_API_KEY: Optional[str] = None
# モデル設定
_MODEL: str = "gpt-3.5-turbo"  # デフォルトモデル


def configure(api_key: str | None = None, model: str = "gpt-3.5-turbo") -> None:
    """
    dariko の設定を行う。

    Args:
        api_key: LLM APIのキー。Noneの場合は環境変数 DARIKO_API_KEY から取得を試みる。
        model: 使用するLLMモデル名。デフォルトは "gpt-3.5-turbo"
    """
    global _API_KEY, _MODEL
    _API_KEY = api_key or os.getenv("DARIKO_API_KEY")
    _MODEL = model


def get_api_key() -> str:
    """設定されたAPIキーを取得する。未設定の場合はエラーを投げる。"""
    if _API_KEY is None:
        raise RuntimeError(
            "APIキーが設定されていません。configure() で設定するか、" "環境変数 DARIKO_API_KEY を設定してください。"
        )
    return _API_KEY


def get_model() -> str:
    """設定されたモデル名を取得する。"""
    return _MODEL
