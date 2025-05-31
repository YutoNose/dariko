import os
from pydantic import BaseModel

from dariko import ask, set_config

# APIキーの設定(環境変数から取得)
llm_key = os.environ.get("DARIKO_API_KEY")
set_config(model="gpt-4o-mini", llm_key=llm_key)


class Person(BaseModel):
    name: str
    age: int
    dummy: bool


def get_person_info() -> Person:
    """人物情報を取得する関数"""
    return ask(
        '以下の形式のJSONを返してください:\n{"name": "山田太郎", "age": 25, "dummy": false}', output_model=Person
    )


# 関数の戻り値として使用
person = get_person_info()
print(f"名前: {person.name}")
print(f"年齢: {person.age}")
print(f"ダミー: {person.dummy}")
