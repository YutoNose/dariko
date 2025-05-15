import os
from typing import Optional


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


def get_api_key() -> str:
    """設定されたAPIキーを取得する。未設定の場合はエラーを投げる。"""
    if _API_KEY is None:
        raise RuntimeError(
            "APIキーが設定されていません。configure() で設定するか、"
            "環境変数 DARIKO_API_KEY を設定してください。"
        )
    return _API_KEY 
