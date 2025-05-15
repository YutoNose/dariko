import ast
import inspect
from typing import Any, Type, get_type_hints, List

from pydantic import BaseModel


def infer_output_model_from_ast(frame) -> type | None:
    """ASTを使用して型アノテーションを推論する"""
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    try:
        with open(filename, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename)
        # 直前のAnnAssign（型付き代入）を探索
        last_ann = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and hasattr(node, "lineno") and node.lineno < lineno:
                if last_ann is None or node.lineno > last_ann.lineno:
                    last_ann = node
        if last_ann and isinstance(last_ann.target, ast.Name):
            varname = last_ann.target.id
            type_str = ast.unparse(last_ann.annotation)
            try:
                model = eval(type_str, frame.f_globals, frame.f_locals)
                if isinstance(model, type) and issubclass(model, BaseModel):
                    return model
                return None
            except Exception:
                return None
    except Exception:
        return None
    return None


def infer_output_model_from_locals(frame) -> Type[Any] | None:
    """ローカル変数の型アノテーションから型を推論する"""
    func_name = frame.f_code.co_name
    if func_name == "<module>":
        # モジュールスコープの場合はASTで推論する
        return infer_output_model_from_ast(frame)
    func = frame.f_globals.get(func_name, None)
    if func is not None:
        hints = get_type_hints(func)
        if len(hints) == 1:
            return next(iter(hints.values()))
    hints = frame.f_locals.get("__annotations__", {})
    if len(hints) == 1:
        return next(iter(hints.values()))
    # fallback: ASTで推論
    return infer_output_model_from_ast(frame)


def infer_output_model_from_return_type(frame) -> Type[Any] | None:
    """呼び出し元関数の戻り値型アノテーションを取得する"""
    try:
        # 呼び出し元の関数オブジェクトを取得
        caller_frame = frame.f_back
        if caller_frame is None:
            return None
        
        # 関数名を取得
        func_name = caller_frame.f_code.co_name
        if func_name == "<module>":
            return None
        
        # 関数オブジェクトを取得
        func = caller_frame.f_locals.get(func_name)
        if func is None:
            return None
        
        # 戻り値型を取得
        hints = get_type_hints(func)
        return hints.get("return")
    except Exception:
        return None


def get_pydantic_model(model):
    """Pydanticモデルを取得する"""
    origin = getattr(model, '__origin__', None)
    if origin in (list, List):
        model = model.__args__[0]
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TypeError("output_modelはPydanticモデルである必要がある")
    return model 
