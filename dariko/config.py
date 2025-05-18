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
    if env_key is not None:
        _API_KEY = env_key
    else:
        raise RuntimeError(
            "APIキーが設定されていません。環境変数 DARIKO_API_KEY を設定してください。"
        )
    _MODEL = model


def get_api_key() -> str:
    """環境変数からAPIキーを取得する。未設定の場合はエラーを投げる。"""
    api_key = os.getenv("DARIKO_API_KEY")
    if api_key is None:
        raise RuntimeError("環境変数 DARIKO_API_KEY が設定されていません。")
    return api_key


def get_model() -> str:
    """設定されたモデル名を取得する。"""
    return _MODEL
