import json
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

    def call(self, messages: List[Dict[str, str]], *, response_schema: Optional[Dict[str, Any]] = None) -> str:
        """Claude API を呼び出して応答テキスト (JSON 文字列) を返す。"""
        if not self.llm_key:
            raise ValueError("APIキーが必要です")

        headers = {
            "x-api-key": self.llm_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
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
