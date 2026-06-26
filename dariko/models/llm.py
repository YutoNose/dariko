from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLM(ABC):
    """LLM プロバイダ実装の基底クラス。

    生成パラメータ (max_tokens / temperature / timeout) を共通で保持する。
    各サブクラスは :meth:`call` を実装し、可能ならば ``response_schema`` を
    使って構造化出力 (JSON) を強制する。
    """

    def __init__(
        self,
        model_name: str,
        llm_key: Optional[str] = None,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.llm_key = llm_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    @abstractmethod
    def call(self, messages: List[Dict[str, str]], *, response_schema: Optional[Dict[str, Any]] = None) -> str:
        """LLM を呼び出し、応答テキスト (JSON 文字列) を返す。

        Args:
            messages: ``{"role", "content"}`` 形式のメッセージ列。
            response_schema: 出力を強制する Pydantic JSON Schema (任意)。
        """
        ...

    @classmethod
    def configure(cls, model_name: str, llm_key: Optional[str] = None, **kwargs: Any) -> "LLM":
        """生成パラメータ込みで LLM インスタンスを生成する。"""
        return cls(model_name=model_name, llm_key=llm_key, **kwargs)
