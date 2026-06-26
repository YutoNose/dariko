from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# python-dotenv は任意依存。未インストールでも動作させる。
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover - 環境依存
    pass


@dataclass
class Config:
    """dariko 全体の実行設定。

    生成パラメータ (max_tokens / temperature / timeout) と、検証失敗時に
    LLM へ再生成を促すリトライ回数 (max_retries) を保持する。
    """

    model: str = "gpt-4o-mini"
    llm_key: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: float = 30.0
    max_retries: int = 2


_config = Config()


def set_config(
    model: Optional[str] = None,
    llm_key: Optional[str] = None,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> None:
    """モデル・APIキー・生成パラメータを設定する。

    指定された引数のみ上書きし、未指定の項目は現在値を保持する。

    Args:
        model: 使用するLLMモデル名 (例: ``gpt-4o-mini``, ``claude-3-opus-20240229``)。
        llm_key: APIキーまたはトークン。
        max_tokens: 生成する最大トークン数。
        temperature: サンプリング温度 (0.0 で決定的)。
        timeout: HTTP リクエストのタイムアウト秒数。
        max_retries: 検証失敗時に LLM へ再生成を促す最大回数。
    """
    if model is not None:
        _config.model = model
    if llm_key is not None:
        _config.llm_key = llm_key
    if max_tokens is not None:
        _config.max_tokens = max_tokens
    if temperature is not None:
        _config.temperature = temperature
    if timeout is not None:
        _config.timeout = timeout
    if max_retries is not None:
        _config.max_retries = max_retries


def get_config() -> Config:
    """現在の設定オブジェクトを返す。"""
    return _config


def get_model() -> str:
    """設定されたモデル名を返す。"""
    return _config.model


def get_llm_key() -> Optional[str]:
    """設定されたLLMキーを返す。"""
    return _config.llm_key
