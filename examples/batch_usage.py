from pydantic import BaseModel
from dariko import ask_batch, configure

# APIキーの設定（環境変数から取得）
configure()  # 環境変数 DARIKO_API_KEY から自動的に取得

# 出力モデルの定義
class Person(BaseModel):
    name: str
    age: int
    dummy: bool
    api_key: str

# バッチ処理
prompts = [
    "以下の形式のJSONを返してください：\n"
    '{"name": "山田太郎", "age": 25, "dummy": false}',
    "以下の形式のJSONを返してください：\n"
    '{"name": "佐藤花子", "age": 30, "dummy": true}'
]

results = ask_batch(prompts, output_model=Person)

# 結果の表示
for i, result in enumerate(results, 1):
    print(f"\n人物 {i}:")
    print(f"名前: {result.name}")
    print(f"年齢: {result.age}")
    print(f"ダミー: {result.dummy}") 
