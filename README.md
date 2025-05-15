# dariko

LLMの出力をPydanticモデルで型安全に扱うためのPythonライブラリ。

## 特徴

- LLMの出力をPydanticモデルで型安全に扱える
- 型アノテーションから自動的に出力モデルを推論
- バッチ処理に対応
- シンプルなAPI

## インストール

```bash
pip install dariko
```

## 使用方法

### 基本的な使い方

```python
from pydantic import BaseModel
from dariko import ask, configure

# APIキーの設定
configure("your-api-key")  # または環境変数 DARIKO_API_KEY を設定

# 出力モデルの定義
class Person(BaseModel):
    name: str
    age: int
    dummy: bool
    api_key: str

# 型アノテーションから自動的にモデルを推論
result: Person = ask("test")
print(result.name)  # "test"
print(result.age)   # 20
print(result.dummy) # True
```

### 明示的にモデルを指定

```python
result = ask("test", output_model=Person)
```

### バッチ処理

```python
from dariko import ask_batch

prompts = ["test1", "test2"]
results = ask_batch(prompts, output_model=Person)
```

## 開発

### セットアップ

```bash
git clone https://github.com/yourusername/dariko.git
cd dariko
pip install -e .
```

### テスト

```bash
pytest tests/
```

## ライセンス

MIT License

```python
from pydantic import BaseModel
from dariko import ask

class Person(BaseModel):
    name: str
    age: int

result: Person = ask("次の JSON を返して: {name:'Alice', age:30}")
print(result)
