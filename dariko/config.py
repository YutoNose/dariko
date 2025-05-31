import os
from typing import Optional

from dotenv import load_dotenv

# .env ファイルを読み込む
load_dotenv()

# グローバル設定
_MODEL: str = "gpt-4o-mini"
_LLM_KEY: Optional[str] = None


def set_config(model: str, llm_key: Optional[str] = None) -> None:
    """
    モデルとLLMキー（APIキーまたはトークン）を設定する

    Args:
        model: 使用するLLMモデル名
        llm_key: APIキーまたはトークン（オプション）
    """
    global _MODEL, _LLM_KEY
    _MODEL = model
    _LLM_KEY = llm_key


def get_model() -> str:
    """設定されたモデル名を返す"""
    return _MODEL


def get_llm_key() -> Optional[str]:
    """設定されたLLMキーを返す"""
    return _LLM_KEY
