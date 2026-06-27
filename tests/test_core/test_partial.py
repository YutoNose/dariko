from dariko.partial import build_partial_model, parse_partial_json
from tests.conftest import Person


def test_parse_partial_json_progressive():
    """途中までの JSON から解釈可能な最大オブジェクトを返す。"""
    assert parse_partial_json('{"name": "test"') == {"name": "test"}
    assert parse_partial_json('{"name": "test", "age":') == {"name": "test"}
    assert parse_partial_json('{"name": "te') == {"name": "te"}
    assert parse_partial_json('{"name": "test", "age": 20, "dummy": true}') == {
        "name": "test",
        "age": 20,
        "dummy": True,
    }


def test_parse_partial_json_empty():
    assert parse_partial_json("") is None
    assert parse_partial_json("   ") is None


def test_build_partial_model_all_optional():
    """部分モデルは全フィールドが省略可能。"""
    partial_cls = build_partial_model(Person)
    obj = partial_cls.model_validate({"name": "x"})  # age / dummy 無しでも通る
    assert obj.name == "x"
    assert obj.age is None
    assert obj.dummy is None
