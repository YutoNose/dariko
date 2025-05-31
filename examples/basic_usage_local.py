import os
from pydantic import BaseModel

from dariko import ask, ask_batch, set_config

# 環境変数からHugging Faceのアクセストークンを取得
llm_key = os.environ.get("DARIKO_API_KEY")
set_config(model="google/gemma-2b", llm_key=llm_key)

# 出力モデル定義
class Person(BaseModel):
    name: str
    age: int
    dummy: bool

prompt = "以下の形式でJSONを返してください:\n" + '{"name": "山田太郎", "age": 25, "dummy": false}'
result: Person = ask(prompt, output_model=Person)
print(result)

# バッチ処理
prompts = [
    "以下の形式でJSONを返してください:\n" + '{"name": "山田太郎", "age": 25, "dummy": false}',
    "以下の形式でJSONを返してください:\n" + '{"name": "佐藤花子", "age": 30, "dummy": true}',
]

results = ask_batch(prompts, output_model=Person)

# 結果表示
for i, result in enumerate(results, 1):
    print(f"\nPerson {i}:")
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Dummy: {result.dummy}")
