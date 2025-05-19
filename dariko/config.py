import os
from typing import Optional

from dotenv import load_dotenv

# .env ファイルを読み込む
load_dotenv()

# APIキー設定
_API_KEY: Optional[str] = None
# モデル設定
_MODEL: str = "gpt-3.5-turbo"  # デフォルトモデル


def configure(model: str = "gpt-3.5-turbo") -> None:
    """
    dariko の設定を行う。

    Args:
        model: 使用するLLMモデル名。デフォルトは "gpt-3.5-turbo"
    """
    global _API_KEY, _MODEL
    env_key = os.getenv("DARIKO_API_KEY")
    if env_key is None:
        raise RuntimeError("APIキーが設定されていません。環境変数 DARIKO_API_KEY を設定してください。")
    _API_KEY = env_key
    _MODEL = model


def get_api_key() -> str:
    """設定されたAPIキーを取得する。未設定の場合はエラーを投げる。"""
    if _API_KEY is None:
        raise RuntimeError("APIキーが設定されていません。configure()を呼び出してください。")
    return _API_KEY


def get_model() -> str:
    """設定されたモデル名を取得する。"""
    return _MODEL
