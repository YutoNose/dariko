import requests
from .llm import LLM

class Claude(LLM):
    def __init__(self, model_name: str, llm_key: str):
        super().__init__(model_name, llm_key)
        self.api_url = "https://api.anthropic.com/v1/messages"

    def call(self, messages):
        if not self.llm_key:
            raise ValueError("APIキーが必要です")
        headers = {
            "x-api-key": self.llm_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        # Claude APIのメッセージ形式に変換
        prompt = self._format_messages(messages)
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post(self.api_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def _format_messages(self, messages):
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages]) 
