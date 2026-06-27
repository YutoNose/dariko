import json
from collections.abc import Iterator
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

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.llm_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self, messages: List[Dict[str, str]], response_schema: Optional[Dict[str, Any]], stream: bool
    ) -> Dict[str, Any]:
        if response_schema is not None:
            response_format: Dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": "dariko_output", "schema": response_schema, "strict": False},
            }
        else:
            response_format = {"type": "json_object"}
        return {
            "model": self.model_name,
            "messages": messages,
            "response_format": response_format,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }

    def call(self, messages: List[Dict[str, str]], *, response_schema: Optional[Dict[str, Any]] = None) -> str:
        """OpenAI API を呼び出して応答テキストを返す。"""
        if not self.llm_key:
            raise ValueError("API key is required for OpenAI models")

        r = requests.post(
            self.api_url,
            headers=self._headers(),
            json=self._payload(messages, response_schema, stream=False),
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI API call failed: {r.text}")
        return r.json()["choices"][0]["message"]["content"]

    def call_stream(
        self, messages: List[Dict[str, str]], *, response_schema: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """OpenAI API を SSE で呼び出し、content の増分を逐次 yield する。"""
        if not self.llm_key:
            raise ValueError("API key is required for OpenAI models")

        with requests.post(
            self.api_url,
            headers=self._headers(),
            json=self._payload(messages, response_schema, stream=True),
            timeout=self.timeout,
            stream=True,
        ) as r:
            if r.status_code != 200:
                raise RuntimeError(f"OpenAI API call failed: {r.text}")
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
