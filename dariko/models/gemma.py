from typing import Dict, List
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .llm import LLM


class Gemma(LLM):
    """Google Gemmaモデル用の実装"""

    def __init__(self, model_name: str, llm_key: str = None):
        super().__init__(model_name=model_name, llm_key=llm_key)
        if not llm_key:
            raise ValueError("Hugging Face token is required for Gemma models")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=llm_key)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16, token=llm_key
        )

    def call(self, messages: List[Dict[str, str]]) -> str:
        """Gemmaモデルを呼び出して応答を取得する"""
        # メッセージをプロンプト形式に変換
        prompt = self._format_messages(messages)

        # トークン化
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成
        outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

        # デコード
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # プロンプト部分を除去
        return response[len(prompt) :]

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """メッセージリストをプロンプト形式に変換"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"{role}: {content}\n"
        return formatted
