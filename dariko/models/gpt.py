import requests
from typing import Dict, List

from .llm import LLM


class GPT(LLM):
    """OpenAIのGPTモデル用の実装"""

    def __init__(self, model_name: str, llm_key: str):
        super().__init__(model_name=model_name, llm_key=llm_key)
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def call(self, messages: List[Dict[str, str]]) -> str:
        """OpenAI APIを呼び出して応答を取得する"""
        if not self.llm_key:
            raise ValueError("API key is required for OpenAI models")

        r = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.llm_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )

        if r.status_code != 200:
            raise RuntimeError(f"OpenAI API call failed: {r.text}")

        return r.json()["choices"][0]["message"]["content"]
