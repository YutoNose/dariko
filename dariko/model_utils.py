from __future__ import annotations

import ast
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Type, get_type_hints

from pydantic import BaseModel

# ロガーの設定
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────
def _validate(model: Any) -> Type[BaseModel] | None:
    """Pydantic Model かどうかを判定し、list[T] にも対応する。BaseModelそのものは除外。"""
    origin = getattr(model, "__origin__", None)
    if origin is list:  # list[T] -> T を取り出す
        model = model.__args__[0]
    # BaseModelそのものは除外
    if model is BaseModel:
        return None
    return model if inspect.isclass(model) and issubclass(model, BaseModel) else None


def _model_from_ast(frame) -> Type[BaseModel] | None:
    """直前行以前の AnnAssign または Assign+type_comment から型を推定。"""
    # 呼び出し元のフレームを取得
    caller_frame = frame.f_back
    if caller_frame is None:
        logger.debug("No caller frame found")
        return None

    # 呼び出し元のファイルをパース
    try:
        file_path = Path(caller_frame.f_code.co_filename)
        logger.debug(f"Parsing file: {file_path}")
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"Failed to parse file: {e}")
        return None

    caller_line = caller_frame.f_lineno
    logger.debug(f"Caller line: {caller_line}")

    # 関数の戻り値の型アノテーションを探す
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.returns is not None:
                try:
                    ann_type_str = ast.unparse(node.returns)
                    logger.debug(f"Found function return type: {ann_type_str}")
                    ann = eval(ann_type_str, caller_frame.f_globals, caller_frame.f_locals)
                    if model := _validate(ann):
                        return model
                except Exception as e:
                    logger.debug(f"Failed to evaluate function return type: {e}")

    # 変数の型アノテーションを探す
    for node in ast.walk(tree):
        node_line = getattr(node, "lineno", 0)
        if node_line > caller_line:
            continue

        # AnnAssign: result: Person = ...
        if isinstance(node, ast.AnnAssign):
            logger.debug(f"Found AnnAssign at line {node_line}")
            try:
                ann_type_str = ast.unparse(node.annotation)
                logger.debug(f"Type annotation string: {ann_type_str}")
                ann = eval(ann_type_str, caller_frame.f_globals, caller_frame.f_locals)
                if model := _validate(ann):
                    return model
            except Exception as e:
                logger.debug(f"Failed to evaluate annotation: {e}")

        # Assign + type_comment: result = ...  # type: Person
        elif isinstance(node, ast.Assign):
            if hasattr(node, "type_comment") and node.type_comment:
                logger.debug(f"Found Assign with type_comment at line {node_line}")
                try:
                    ann = eval(node.type_comment, caller_frame.f_globals, caller_frame.f_locals)
                    if model := _validate(ann):
                        return model
                except Exception as e:
                    logger.debug(f"Failed to evaluate type comment: {e}")

    logger.debug("No suitable type annotation found")
    return None


# ─────────────────────────────────────────────────────────────
# パブリック API
# ─────────────────────────────────────────────────────────────
def infer_output_model(frame=None) -> Type[BaseModel] | None:
    """
    実行中フレームから Pydantic モデル型を推定するユーティリティ。
    優先順位:
        1. 呼び出し元関数の return 型ヒント（関数オブジェクト or AST）
        2. 現フレームのローカル変数アノテーション
        3. AST 解析による推定
    """
    import ast
    import inspect

    # ------------------------------------------------------------
    # ❶ 呼び出し側から frame が渡された場合はそれを最優先
    # ------------------------------------------------------------
    if frame is not None:
        user_frame = frame
    else:
        # ❷ さもなくば「site-packages 内の dariko」を除いた最初のフレーム
        stack = inspect.stack()
        for s in stack:
            path = os.path.abspath(s.filename)
            # site-packages or editable install の dariko パッケージだけを除外
            if path.endswith("dariko/llm.py") or "/site-packages/dariko/" in path:
                continue
            user_frame = s.frame
            break
        else:
            user_frame = stack[-1].frame

    frame = user_frame
    logger.debug(f"Current frame: {frame.f_code.co_name} at line {frame.f_lineno} " f"in {frame.f_code.co_filename}")

    # 0) 現在実行中の関数 get_person_info() の return 型を調べる
    if frame.f_code.co_name != "<module>":
        try:
            # 関数オブジェクトを frame から解決
            func_obj = frame.f_locals.get(frame.f_code.co_name) or frame.f_globals.get(frame.f_code.co_name)
            if func_obj:
                return_type = get_type_hints(func_obj).get("return")
                logger.debug(f"Current func return type: {return_type}")
                if model := _validate(return_type):
                    return model

            # --- AST fallback ---
            file_path = Path(frame.f_code.co_filename)
            with file_path.open(encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == frame.f_code.co_name:
                    if node.returns is not None:
                        ann_type_str = ast.unparse(node.returns)
                        logger.debug(f"AST: current func return annotation: {ann_type_str}")
                        try:
                            ann = eval(ann_type_str, frame.f_globals, frame.f_locals)
                            if model := _validate(ann):
                                return model
                        except Exception as e:
                            logger.debug(f"Failed to evaluate annotation: {e}")
                            continue
        except Exception as e:
            logger.debug(f"Failed to inspect current function: {e}")

    # 1) AST 解析による推定（変数の型アノテーションを含む）
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
