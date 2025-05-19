# dariko

LLMの出力をPydanticモデルで型安全に扱うためのPythonライブラリ。

## 特徴

- LLMの出力をPydanticモデルで型安全に扱える
- 型アノテーションから自動的に出力モデルを推論
- バッチ処理に対応
- シンプルなAPI
- 環境変数から自動的にAPIキーを読み込み

## インストール

```bash
pip install dariko
```

## 使用方法

### 基本的な使い方

```python
from pydantic import BaseModel
from dariko import ask

# 出力モデルの定義
class Person(BaseModel):
    name: str
    age: int
    dummy: bool

# 型アノテーションから自動的にモデルを推論
result: Person = ask("以下の形式のJSONを返してください:\n" + '{"name": "山田太郎", "age": 25, "dummy": false}')
print(result.name)  # "山田太郎"
print(result.age)   # 25
print(result.dummy) # False
```

### 明示的にモデルを指定

```python
result = ask("test", output_model=Person)
```

### バッチ処理

```python
from dariko import ask_batch

prompts = [
    "以下の形式のJSONを返してください:\n" + '{"name": "山田太郎", "age": 25, "dummy": false}',
    "以下の形式のJSONを返してください:\n" + '{"name": "佐藤花子", "age": 30, "dummy": true}',
]

results = ask_batch(prompts, output_model=Person)

# 結果の表示
for i, result in enumerate(results, 1):
    print(f"\n人物 {i}:")
    print(f"名前: {result.name}")
    print(f"年齢: {result.age}")
    print(f"ダミー: {result.dummy}")
```

## 型推論の仕組み

Darikoは以下の優先順位で型を推論します：

1. 呼び出し元関数のreturn型ヒント
2. 現フレームのローカル変数アノテーション（1個だけの場合）
3. AST解析による推定

詳細な実装については、[examples/logic.md](examples/logic.md)を参照してください。

## 環境変数

以下の環境変数を設定することで、Darikoの動作を制御できます：

- `DARIKO_API_KEY`: APIキー（必須）
- `DARIKO_MODEL`: 使用するモデル名（デフォルト: "gpt-4"）

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
