"""ストリーミング中の不完全な JSON を逐次パースするためのユーティリティ。

部分的に届いた JSON 文字列を「閉じられる範囲で」補完してパースし、
全フィールドを Optional 化した「部分モデル」へ検証する。
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional, Type

from pydantic import BaseModel, create_model


def _balance(prefix: str) -> Optional[str]:
    """JSON のプレフィックスを、開いた文字列・括弧を閉じて parse 可能な形にする。

    括弧の対応が壊れている (閉じ過ぎ) 場合は ``None`` を返す。
    """
    in_str = False
    esc = False
    stack: list[str] = []
    for ch in prefix:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if not stack:
                    return None
                stack.pop()
    out = prefix
    if in_str:
        out += '"'
    out += "".join(reversed(stack))
    return out


def parse_partial_json(buffer: str) -> Optional[Any]:
    """部分的な JSON 文字列から、解釈可能な最大のオブジェクトを返す。

    末尾の不完全なトークン (途中のキーや値) を切り詰めながら、最初に
    parse できたものを返す。何も解釈できなければ ``None``。
    """
    s = buffer.strip()
    if not s:
        return None
    # 長いプレフィックスから順に、閉じて parse できるものを探す
    for end in range(len(s), 0, -1):
        balanced = _balance(s[:end])
        if balanced is None:
            continue
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            continue
    return None


@lru_cache(maxsize=128)
def build_partial_model(model: Type[BaseModel]) -> Type[BaseModel]:
    """全フィールドを Optional (既定値 None) にした部分モデルを生成する。

    ストリーミング途中の「まだ揃っていない」データを検証するために使う。
    """
    fields: dict[str, Any] = {
        name: (Optional[field.annotation], None) for name, field in model.model_fields.items()
    }
    return create_model(f"Partial{model.__name__}", **fields)
