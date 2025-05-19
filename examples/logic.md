# Darikoの型推論システムの実装

## 概要

Darikoは、Pythonの型アノテーションを活用して、LLMからの出力を自動的にPydanticモデルに変換する機能を提供します。このドキュメントでは、その実装の詳細と、特にAST（抽象構文木）を用いた型推論の仕組みについて説明します。

## 型推論の優先順位

Darikoは以下の優先順位で型を推論します：

1. 呼び出し元関数のreturn型ヒント
2. 変数アノテーション（AnnAssign/type_comment）
3. AST解析による推定

## 実践例

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

## ASTによる型推論の詳細

### 1. ASTとは

AST（Abstract Syntax Tree）は、プログラムのソースコードを木構造で表現したものです。Pythonの`ast`モジュールを使用して、ソースコードを解析し、その構造を理解することができます。

### 2. 型アノテーションの検出方法

Darikoは以下のパターンで型アノテーションを検出します：

- 関数の戻り値型アノテーション（`def func() -> Model:`）
- 変数の型アノテーション（`result: Model = ...` または `# type: Model`）

#### 2.1 型アノテーション付き代入（AnnAssign）

```python
result: Person = ask(prompt)
```

#### 2.2 型コメント付き代入（Assign + type_comment）

```python
result = ask(prompt)  # type: Person
```

#### 2.3 関数の戻り値型アノテーション

```python
def get_person() -> Person:
    return ask(...)
```

### 3. 実装の流れ

- 呼び出し元のフレーム情報から、該当ファイルをASTでパース
- 関数定義や変数アノテーションを探索し、型文字列を抽出
- `eval`で型オブジェクトに変換し、PydanticのBaseModelサブクラスか検証
- 最初に見つかった有効な型を返す

### 4. デバッグとロギング

実装では、詳細なデバッグ情報を提供するために、以下のようなログ出力を行っています：

```python
logger.debug(f"Parsing file: {file_path}")
logger.debug(f"Caller line: {caller_line}")
logger.debug(f"Found function return type: {ann_type_str}")
```

これにより、型推論の過程を追跡し、問題が発生した場合の原因特定が容易になります。

## 注意点・制限事項

- 型アノテーションが取得できない場合は `output_model` を明示的に指定してください。
- 型推論は「関数の戻り値型」→「変数アノテーション」→「AST解析」の順で行われます。
- 型アノテーションはPydanticのBaseModelサブクラスである必要があります。
- 型アノテーションは、`ask`関数の呼び出しと同じ行か、直前の行に存在する必要があります
- 複数の型アノテーションが存在する場合、最も近いものが使用されます
- list[Model] 形式にも対応しています

## 今後の改善点

1. より複雑な型アノテーションパターンのサポート
2. 型推論の精度向上
3. エラーメッセージの改善
4. パフォーマンスの最適化
