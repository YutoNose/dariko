from typing import Any, Dict, List, Optional

from .llm import LLM


class Gemma(LLM):
    """Google Gemma (ローカル推論) 用の実装。

    ``torch`` / ``transformers`` は重量級依存のため、Gemma を実際に使う時だけ
    遅延インポートする。未インストールの場合は導入方法を案内する。
    """

    def __init__(self, model_name: str, llm_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model_name=model_name, llm_key=llm_key, **kwargs)
        if not llm_key:
            raise ValueError("Hugging Face token is required for Gemma models")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Gemma を使うには追加依存が必要です: `pip install dariko[gemma]`"
            ) from e

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=llm_key)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16, token=llm_key
        )

    def call(self, messages: List[Dict[str, str]], *, response_schema: Optional[Dict[str, Any]] = None) -> str:
        """Gemma モデルを呼び出して応答テキストを返す。"""
        prompt = self._format_messages(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # プロンプト部分を除去
        return response[len(prompt):]

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """メッセージリストをプロンプト形式に変換する。"""
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role']}: {msg['content']}\n"
        return formatted
