from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path
from typing import Any, get_type_hints, Type
import os

from pydantic import BaseModel

# ロガーの設定
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────
def _validate(model: Any) -> Type[BaseModel] | None:
    """Pydantic Model かどうかを判定し、list[T] にも対応する."""
    origin = getattr(model, "__origin__", None)
    if origin is list:                     # list[T] -> T を取り出す
        model = model.__args__[0]
    return model if inspect.isclass(model) and issubclass(model, BaseModel) else None


def _model_from_ast(frame) -> Type[BaseModel] | None:
    """直前行以前の AnnAssign または Assign+type_comment から型を推定。"""
    try:
        file_path = Path(frame.f_code.co_filename)
        logger.debug(f"Parsing file: {file_path}")
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"Failed to parse file: {e}")
        return None

    current_line = frame.f_lineno
    logger.debug(f"Current line: {current_line}")

    var_name = "result"
    logger.debug(f"Looking for AnnAssign/Assign node for variable: {var_name}")

    # ASTの全ノードをデバッグ出力
    for node in ast.walk(tree):
        logger.debug(f"AST node: {ast.dump(node)}")

    target_node = None
    ann_type_str = None
    for node in ast.walk(tree):
        node_line = getattr(node, "lineno", 0)
        logger.debug(f"Checking node_line={node_line} vs current_line={current_line}")
        if node_line > current_line:
            continue
        # AnnAssign: result: Person = ...
        if isinstance(node, ast.AnnAssign):
            if hasattr(node.target, 'id') and node.target.id == var_name:
                logger.debug(f"Matched AnnAssign for {var_name} at line {node_line}")
                if target_node is None or node_line > target_node.lineno:
                    target_node = node
                    ann_type_str = ast.unparse(node.annotation)
        # Assign + type_comment: result = ...  # type: Person
        elif isinstance(node, ast.Assign):
            if hasattr(node.targets[0], 'id') and node.targets[0].id == var_name:
                if hasattr(node, 'type_comment') and node.type_comment:
                    logger.debug(f"Matched Assign+type_comment for {var_name} at line {node_line}")
                    if target_node is None or node_line > target_node.lineno:
                        target_node = node
                        ann_type_str = node.type_comment
    if not target_node or not ann_type_str:
        logger.debug("No suitable AnnAssign/Assign node found")
        return None

    logger.debug(f"Selected node at line {getattr(target_node, 'lineno', None)} for variable: {var_name}")
    logger.debug(f"Type annotation string: {ann_type_str}")

    try:
        ann = eval(ann_type_str, frame.f_globals, frame.f_locals)
        logger.debug(f"Evaluated annotation: {ann}")
        validated = _validate(ann)
        logger.debug(f"Validated model: {validated}")
        return validated
    except Exception as e:
        logger.debug(f"Failed to evaluate annotation: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# パブリック API
# ─────────────────────────────────────────────────────────────
def infer_output_model(frame=None) -> Type[BaseModel] | None:
    """
    実行中フレームから Pydantic モデル型を推定するユーティリティ。
    優先順位:
        1. 呼び出し元関数の return 型ヒント
        2. 現フレームのローカル変数アノテーション（1個だけの場合）
        3. AST 解析による推定
    """
    import inspect, os
    stack = inspect.stack()
    user_frame = None
    for s in stack:
        # darikoパッケージ外のファイルを探す
        if "dariko" not in os.path.abspath(s.filename):
            user_frame = s.frame
            break
    if user_frame is None:
        # fallback: 最後のフレーム
        user_frame = stack[-1].frame

    frame = user_frame
    logger.debug(f"Current frame: {frame.f_code.co_name} at line {frame.f_lineno} in {frame.f_code.co_filename}")

    # 1) 呼び出し元関数の return 型
    caller = frame.f_back
    if caller and caller.f_code.co_name != "<module>":
        logger.debug(f"Checking caller function: {caller.f_code.co_name}")
        func_obj = caller.f_locals.get(caller.f_code.co_name)
        if func_obj:
            return_type = get_type_hints(func_obj).get("return")
            logger.debug(f"Caller return type: {return_type}")
            if (model := _validate(return_type)):
                return model

    # 2) 現フレームのローカル変数アノテーション
    local_hints = frame.f_locals.get("__annotations__", {})
    logger.debug(f"Local annotations: {local_hints}")
    if len(local_hints) == 1:
        if (model := _validate(next(iter(local_hints.values())))):
            return model

    # 3) AST から推定
    logger.debug("Attempting AST inference")
    return _model_from_ast(frame)


def get_pydantic_model(model: Type[Any]) -> Type[BaseModel]:
    """
    与えられた型が Pydantic モデル（あるいは list[T] 形式で T が Pydantic
    モデル）かどうかを確認し、適切でなければ TypeError を投げる。
    """
    validated = _validate(model)
    if validated is None:
        raise TypeError("output_model must be a Pydantic model (or list[Model]).")
    return validated
