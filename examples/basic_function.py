from pydantic import BaseModel

from dariko import ask, configure

# APIキーの設定(環境変数から取得)
configure(model="gpt-4o-mini")  # 環境変数 DARIKO_API_KEY から自動的に取得


class Person(BaseModel):
    name: str
    age: int
    dummy: bool


def get_person_info() -> Person:
    """人物情報を取得する関数"""
    return ask('以下の形式のJSONを返してください:\n{"name": "山田太郎", "age": 25, "dummy": false}')


# 関数の戻り値として使用
person = get_person_info()
print(f"名前: {person.name}")
print(f"年齢: {person.age}")
print(f"ダミー: {person.dummy}")
