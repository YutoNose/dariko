# dariko

LLMの出力をPydanticモデルで型安全に扱うためのPythonライブラリ。

## 特徴

- LLMの出力をPydanticモデルで型安全に扱える
- 型アノテーションから自動的に出力モデルを推論
- バッチ処理に対応
- シンプルなAPI
- 環境変数から自動的にAPIキーを読み込み
- 複数のLLM（GPT, Claude, Gemma等）に対応

## インストール

```bash
pip install dariko
```

## 使用方法

### 基本的な使い方

```python
import os
from pydantic import BaseModel
from dariko import ask, set_config

# APIキーの設定（環境変数から取得）
llm_key = os.environ.get("DARIKO_API_KEY")
set_config(model="gpt-4o-mini", llm_key=llm_key)

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

### ローカルモデル（Gemma）の使用例

```python
import os
from pydantic import BaseModel
from dariko import ask, set_config

# Hugging Faceのアクセストークンを設定
llm_key = os.environ.get("DARIKO_API_KEY")
set_config(model="google/gemma-2b", llm_key=llm_key)

class Person(BaseModel):
    name: str
    age: int
    dummy: bool

result: Person = ask("以下の形式のJSONを返してください:\n" + '{"name": "山田太郎", "age": 25, "dummy": false}')
print(result)
```

### Claudeモデルの使用例

```python
import os
from pydantic import BaseModel
from dariko import ask, set_config

# AnthropicのAPIキーを設定
llm_key = os.environ.get("DARIKO_API_KEY")
set_config(model="claude-3-opus-20240229", llm_key=llm_key)

class Person(BaseModel):
    name: str
    age: int
    dummy: bool

result: Person = ask("以下の形式のJSONを返してください:\n" + '{"name": "山田太郎", "age": 25, "dummy": false}')
print(result)
```

## 型推論の実践例

### 関数の戻り値型アノテーションによる推論

```python
def get_person() -> Person:
    return ask('以下の形式のJSONを返してください:\n{"name": "山田太郎", "age": 25, "dummy": false}')

person = get_person()
print(person.name)  # "山田太郎"
```

### 変数アノテーションによる推論

```python
result: Person = ask('以下の形式のJSONを返してください:\n{"name": "佐藤花子", "age": 30, "dummy": true}')
print(result.name)  # "佐藤花子"
```

### バッチ処理でも型推論が効く

```python
from typing import List

def get_people() -> List[Person]:
    prompts = [
        '以下の形式のJSONを返してください:\n{"name": "山田太郎", "age": 25, "dummy": false}',
        '以下の形式のJSONを返してください:\n{"name": "佐藤花子", "age": 30, "dummy": true}',
    ]
    return ask_batch(prompts)

people = get_people()
for p in people:
    print(p.name)
```

### 注意点
- 型アノテーションが取得できない場合は `output_model` を明示的に指定してください。
- 型推論は「関数の戻り値型」→「変数アノテーション」→「AST解析」の順で行われます。
- 型アノテーションはPydanticのBaseModelサブクラスである必要があります。

## 型推論の仕組み

Darikoは以下の優先順位で型を推論します：

1. 呼び出し元関数のreturn型ヒント
2. 現フレームのローカル変数アノテーション（1個だけの場合）
3. AST解析による推定

詳細な実装については、[examples/logic.md](examples/logic.md)を参照してください。

## 環境変数

以下の環境変数を設定することで、Darikoの動作を制御できます：

- `DARIKO_API_KEY`: APIキー（必須）
  - OpenAI APIキー
  - Anthropic APIキー
  - Hugging Faceアクセストークン
- `DARIKO_MODEL`: 使用するモデル名（デフォルト: "gpt-4"）
  - OpenAIモデル: "gpt-4", "gpt-3.5-turbo" など
  - Claudeモデル: "claude-3-opus-20240229", "claude-3-sonnet-20240229" など
  - Gemmaモデル: "google/gemma-2b" など

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

### リリースプロセス

1. 変更をコミットしてプルリクエストを作成：
```bash
./scripts/release.sh
```

2. スクリプトの実行手順：
   - コミットタイプを選択（新機能/バグ修正/破壊的変更）
   - 変更内容を入力
   - 破壊的変更の場合は詳細を入力

3. バージョン管理の仕組み：
   - コミットメッセージに基づいて自動的にバージョンが更新されます
   - `feat:` → マイナーバージョンアップ（0.1.0 → 0.2.0）
   - `fix:` → パッチバージョンアップ（0.1.0 → 0.1.1）
   - `BREAKING CHANGE:` → メジャーバージョンアップ（0.1.0 → 1.0.0）

4. リリースの流れ：
   - プルリクエストが作成されます
   - レビュー後にマージ
   - マージされると自動的に：
     - バージョンが更新
     - GitHubリリースが作成
     - PyPIにパッケージがアップロード

5. 注意点：
   - コミットメッセージは[Angularのコミットメッセージ規約](https://www.conventionalcommits.org/ja/v1.0.0/)に従ってください
   - 複数のコミットがある場合、最も大きな変更に基づいてバージョンが更新されます
   - GitHub CLI（`gh`）のインストールと認証が必要です

## ライセンス

MIT License
