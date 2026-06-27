import json
from collections.abc import Iterator
from typing import Any, Dict, List, Optional

import requests

from .llm import LLM


class Claude(LLM):
    """Anthropic Claude モデル用の実装。

    ``response_schema`` が渡された場合は tool-use (関数呼び出し) を強制し、
    確実に JSON (スキーマ準拠の構造化データ) を取得する。スキーマが無い
    場合はテキスト応答をそのまま返す。
    """

    def __init__(self, model_name: str, llm_key: str, **kwargs: Any):
        super().__init__(model_name=model_name, llm_key=llm_key, **kwargs)
        self.api_url = "https://api.anthropic.com/v1/messages"

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.llm_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def call(self, messages: List[Dict[str, str]], *, response_schema: Optional[Dict[str, Any]] = None) -> str:
        """Claude API を呼び出して応答テキスト (JSON 文字列) を返す。"""
        if not self.llm_key:
            raise ValueError("APIキーが必要です")

        headers = self._headers()
        prompt = self._format_messages(messages)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_schema is not None:
            # tool-use で JSON 出力を強制する
            payload["tools"] = [
                {
                    "name": "dariko_output",
                    "description": "スキーマに従った構造化データを返す",
                    "input_schema": response_schema,
                }
            ]
            payload["tool_choice"] = {"type": "tool", "name": "dariko_output"}

        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return self._extract_content(resp.json())

    def call_stream(
        self, messages: List[Dict[str, str]], *, response_schema: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """Claude API を SSE で呼び出し、text の増分を逐次 yield する。

        ストリーミング時は tool-use を使わず、テキスト (JSON 文字列) を逐次受け取る。
        """
        if not self.llm_key:
            raise ValueError("APIキーが必要です")

        prompt = self._format_messages(messages)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        with requests.post(
            self.api_url, headers=self._headers(), json=payload, timeout=self.timeout, stream=True
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: ") :])
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield delta["text"]

    @staticmethod
    def _extract_content(body: Dict[str, Any]) -> str:
        """応答 body から JSON 文字列を取り出す。

        tool_use ブロックがあればその input を、無ければ最初の text ブロックを返す。
        """
        for block in body.get("content", []):
            if block.get("type") == "tool_use":
                return json.dumps(block["input"])
        for block in body.get("content", []):
            if block.get("type", "text") == "text" and "text" in block:
                return block["text"]
        raise RuntimeError(f"Claude API から有効な応答を取得できませんでした: {body}")

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """メッセージリストを単一プロンプト文字列に変換する。"""
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
