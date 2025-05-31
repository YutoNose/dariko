# dariko.models - LLM拡張用ディレクトリ

このディレクトリは、darikoパッケージで利用する各種LLM（大規模言語モデル）の実装・拡張を管理するためのものです。

## ディレクトリ構成

- `llm.py`   : LLMの抽象基底クラス（Base Class）
- `gpt.py`   : OpenAI GPT系API用の実装
- `gemma.py` : Google Gemma等、ローカル/OSSモデル用の実装
- `claude.py`: Anthropic Claude API用の実装
- `__init__.py` : モジュールエクスポート

## 設計方針

- **LLMの追加・切り替えを容易にするため、抽象基底クラス（`LLM`）を用意しています。**
- 各モデルごとにサブクラスを作成し、`call()`メソッドを実装してください。
- APIキーやトークンは`llm_key`として統一的に扱います。
- モデルの初期化は`configure`クラスメソッドで行います。

## 実装例

### 1. 抽象基底クラス（llm.py）

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class LLM(ABC):
    def __init__(self, model_name: str, llm_key: Optional[str] = None):
        self.model_name = model_name
        self.llm_key = llm_key

    @abstractmethod
    def call(self, messages: List[Dict[str, str]]) -> str:
        """LLMにプロンプトを投げて応答を返す。"""
        pass

    @classmethod
    def configure(cls, model_name: str, llm_key: Optional[str] = None) -> 'LLM':
        """モデル名・キーでインスタンス生成"""
        return cls(model_name, llm_key)
```

### 2. OpenAI GPT系APIの実装例（gpt.py）

```python
import requests
from .llm import LLM

class GPT(LLM):
    def __init__(self, model_name: str, llm_key: str):
        super().__init__(model_name, llm_key)
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def call(self, messages):
        if not self.llm_key:
            raise ValueError("APIキーが必要です")
        headers = {"Authorization": f"Bearer {self.llm_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "messages": messages}
        resp = requests.post(self.api_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

### 3. ローカルモデルの実装例（gemma.py）

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .llm import LLM

class Gemma(LLM):
    def __init__(self, model_name: str, llm_key: str = None):
        super().__init__(model_name, llm_key)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=llm_key)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=llm_key)

    def call(self, messages):
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _format_messages(self, messages):
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
```

### 4. Anthropic Claude APIの実装例（claude.py）

```python
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
```

## 新しいモデルの追加方法

1. `llm.py`の`LLM`を継承した新しいクラスを作成。
2. `call(self, messages)`を実装。
3. 必要に応じて`configure`クラスメソッドをオーバーライド。
4. `dariko/llm.py`の`MODEL_MAPPING`にモデル名プレフィックスとクラスを追加。

## 注意事項

- APIキーやトークンは絶対にハードコーディングせず、環境変数や引数で渡すこと。
- モデルの追加時は、テストコードも必ず追加・修正してください。
- Hugging Face等のローカルモデルは、アクセストークンが必要な場合があります。

## 参考
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)

---

何か不明点があれば、`dariko`リポジトリのIssueやドキュメントを参照してください。 
