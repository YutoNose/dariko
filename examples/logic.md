# Darikoの型推論システムの実装

## 概要

Darikoは、Pythonの型アノテーションを活用して、LLMからの出力を自動的にPydanticモデルに変換する機能を提供します。このドキュメントでは、その実装の詳細と、特にAST（抽象構文木）を用いた型推論の仕組みについて説明します。

## 型推論の優先順位

Darikoは以下の優先順位で型を推論します：

1. 呼び出し元関数のreturn型ヒント
2. 現フレームのローカル変数アノテーション（1個だけの場合）
3. AST解析による推定

## ASTによる型推論の詳細

### 1. ASTとは

AST（Abstract Syntax Tree）は、プログラムのソースコードを木構造で表現したものです。Pythonの`ast`モジュールを使用して、ソースコードを解析し、その構造を理解することができます。

### 2. 型アノテーションの検出方法

Darikoは以下の2つのパターンで型アノテーションを検出します：

#### 2.1 型アノテーション付き代入（AnnAssign）

```python
result: Person = ask(prompt)  # このような形式
```

この場合、ASTでは以下のような構造になります：

```python
AnnAssign(
    target=Name(id='result', ctx=Store()),
    annotation=Name(id='Person', ctx=Load()),
    value=Call(...),
    simple=1
)
```

#### 2.2 型コメント付き代入（Assign + type_comment）

```python
result = ask(prompt)  # type: Person  # このような形式
```

この場合、ASTでは以下のような構造になります：

```python
Assign(
    targets=[Name(id='result', ctx=Store())],
    value=Call(...),
    type_comment='Person'
)
```

### 3. 実装の詳細

#### 3.1 ファイルの解析

```python
file_path = Path(frame.f_code.co_filename)
tree = ast.parse(file_path.read_text(encoding="utf-8"))
```

- 現在実行中のフレームからファイルパスを取得
- ファイルの内容を読み込み
- `ast.parse()`でASTを生成

#### 3.2 ノードの探索

```python
for node in ast.walk(tree):
    node_line = getattr(node, "lineno", 0)
    if node_line > current_line:
        continue
```

- `ast.walk()`で全ノードを走査
- 現在の行より後のノードは無視
- 各行のノードを評価

#### 3.3 型アノテーションの抽出

```python
if isinstance(node, ast.AnnAssign):
    if hasattr(node.target, 'id') and node.target.id == var_name:
        ann_type_str = ast.unparse(node.annotation)
elif isinstance(node, ast.Assign):
    if hasattr(node.targets[0], 'id') and node.targets[0].id == var_name:
        if hasattr(node, 'type_comment') and node.type_comment:
            ann_type_str = node.type_comment
```

- `AnnAssign`ノードの場合：`annotation`属性から型を取得
- `Assign`ノードの場合：`type_comment`属性から型を取得

#### 3.4 型の評価

```python
ann = eval(ann_type_str, frame.f_globals, frame.f_locals)
validated = _validate(ann)
```

- 型文字列を実際の型オブジェクトに評価
- グローバルとローカルの名前空間を使用
- Pydanticモデルとして妥当か検証

### 4. デバッグとロギング

実装では、詳細なデバッグ情報を提供するために、以下のようなログ出力を行っています：

```python
logger.debug(f"Parsing file: {file_path}")
logger.debug(f"Current line: {current_line}")
logger.debug(f"AST node: {ast.dump(node)}")
```

これにより、型推論の過程を追跡し、問題が発生した場合の原因特定が容易になります。

## 使用例

```python
from pydantic import BaseModel
from dariko import ask

class Person(BaseModel):
    name: str
    age: int

# 型アノテーションを使用した呼び出し
result: Person = ask("名前と年齢を教えてください")
print(result.name)  # 型安全にアクセス可能
```

## 制限事項

1. 型アノテーションは、`ask`関数の呼び出しと同じ行か、直前の行に存在する必要があります
2. 複数の型アノテーションが存在する場合、最も近いものが使用されます
3. 型アノテーションは、Pydanticモデルまたは`list[PydanticModel]`の形式である必要があります

## 今後の改善点

1. より複雑な型アノテーションパターンのサポート
2. 型推論の精度向上
3. エラーメッセージの改善
4. パフォーマンスの最適化 
