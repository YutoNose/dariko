from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type


class LLM(ABC):
    """LLMの基底クラス"""

    def __init__(self, model_name: str, llm_key: Optional[str] = None):
        self.model_name = model_name
        self.llm_key = llm_key

    @abstractmethod
    def call(self, messages: List[Dict[str, str]]) -> str:
        """LLMを呼び出して応答を取得する"""
        pass

    @classmethod
    def configure(cls, model_name: str, llm_key: Optional[str] = None) -> "LLM":
        """LLMインスタンスを設定する"""
        return cls(model_name=model_name, llm_key=llm_key)
