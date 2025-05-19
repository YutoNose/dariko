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

## フレームと型推論

### フレームとは

フレームは、Pythonの実行時における「関数呼び出しのスタック」の各段階を表すオブジェクトです。  
具体的には、関数が呼び出されるたびに、その関数のローカル変数や実行位置などの情報が「フレーム」としてスタックに積まれます。

### フレームの主な役割
- **ローカル変数の管理**: 関数内で定義された変数や引数が格納される
- **実行位置の記録**: 現在実行中の行番号やファイル名を保持
- **呼び出し元の情報**: `frame.f_back` で呼び出し元のフレームを参照可能

### 具体例
例えば、以下のようなコードがあるとします：

```python
def outer():
    x = 1
    inner()

def inner():
    y = 2
    # ここで frame を取得すると...
    frame = inspect.currentframe()
    print(frame.f_locals)  # {'y': 2}
    print(frame.f_back.f_locals)  # {'x': 1, 'inner': <function inner>}
```

- `inner()` 内で `frame` を取得すると、`inner` のローカル変数（`y`）や実行位置が記録されています。
- `frame.f_back` で呼び出し元（`outer`）のフレームを参照でき、`outer` のローカル変数（`x`）も取得できます。

### Darikoでの利用
Darikoでは、`ask` 関数が呼び出された際に、  
- 呼び出し元のフレーム（`frame.f_back`）から、ユーザーコードのファイルパスや行番号を取得
- その情報を元にASTを解析し、型アノテーションを探す

という流れで型推論を行っています。

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

1. **呼び出し元のフレーム情報を取得**
   - `frame.f_back` で呼び出し元のフレームを取得
   - 呼び出し元のファイルパス（`frame.f_code.co_filename`）と行番号（`frame.f_lineno`）を記録

2. **ASTによるファイル解析**
   - 呼び出し元のファイルを読み込み、`ast.parse()` でASTを生成
   - デバッグログでファイルパスや行番号を出力

3. **関数の戻り値型アノテーションの探索**
   - `ast.walk()` で全ノードを走査
   - `ast.FunctionDef` ノードを探し、`returns` 属性から型アノテーションを抽出
   - `ast.unparse()` で型文字列を取得し、`eval()` で型オブジェクトに変換
   - `_validate()` でPydanticのBaseModelサブクラスか検証

4. **変数の型アノテーションの探索**
   - 呼び出し元の行番号より前のノードのみを対象に探索
   - `ast.AnnAssign` ノード（`result: Person = ...`）から型アノテーションを抽出
   - `ast.Assign` ノードの `type_comment` 属性（`# type: Person`）から型アノテーションを抽出
   - 同様に `eval()` と `_validate()` で型を検証

5. **型の検証と返却**
   - `_validate()` 関数で、型がPydanticのBaseModelサブクラスか確認
   - `list[T]` 形式の場合は `T` を取り出して検証
   - 最初に見つかった有効な型を返す

6. **エラーハンドリングとロギング**
   - 各ステップで発生する例外をキャッチし、デバッグログに出力
   - 型アノテーションが取得できない場合は `None` を返す

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
