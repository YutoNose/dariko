from typing import Any, Dict, List, Optional

import requests

from .llm import LLM


class GPT(LLM):
    """OpenAI の GPT モデル用の実装。

    ``response_schema`` が渡された場合は Structured Outputs
    (``response_format: json_schema``) で出力を強制し、無い場合は
    ``json_object`` モードにフォールバックする。
    """

    def __init__(self, model_name: str, llm_key: str, **kwargs: Any):
        super().__init__(model_name=model_name, llm_key=llm_key, **kwargs)
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def call(self, messages: List[Dict[str, str]], *, response_schema: Optional[Dict[str, Any]] = None) -> str:
        """OpenAI API を呼び出して応答テキストを返す。"""
        if not self.llm_key:
            raise ValueError("API key is required for OpenAI models")

        if response_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "dariko_output",
                    "schema": response_schema,
                    "strict": False,
                },
            }
        else:
            response_format = {"type": "json_object"}

        r = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.llm_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "response_format": response_format,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            timeout=self.timeout,
        )

        if r.status_code != 200:
            raise RuntimeError(f"OpenAI API call failed: {r.text}")

        return r.json()["choices"][0]["message"]["content"]
