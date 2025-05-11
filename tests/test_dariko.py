import os
from typing import List

import pytest
from pydantic import BaseModel

from dariko import ask, ask_batch, configure, ValidationError


class Person(BaseModel):
    name: str
    age: int


def test_configure():
    # 環境変数から設定
    os.environ["DARIKO_API_KEY"] = "test_key"
    configure()
    result: Person = ask("test")
    assert result.dummy is True
    assert result.api_key == "test_key"

    # 直接設定
    configure("direct_key")
    result: Person = ask("test")
    assert result.api_key == "direct_key"


def test_ask_with_variable_annotation():
    result: Person = ask("test")
    assert isinstance(result, Person)
    assert result.dummy is True


def test_ask_with_return_type():
    def get_person(prompt: str) -> Person:
        return ask(prompt)

    result = get_person("test")
    assert isinstance(result, Person)
    assert result.dummy is True


def test_ask_with_explicit_model():
    result = ask("test", output_model=Person)
    assert isinstance(result, Person)
    assert result.dummy is True


def test_ask_batch():
    prompts = ["test1", "test2"]
    results: List[Person] = ask_batch(prompts)
    assert len(results) == 2
    assert all(isinstance(r, Person) for r in results)
    assert all(r.dummy is True for r in results)


def test_validation_error():
    with pytest.raises(ValidationError):
        result: Person = ask("invalid")  # ダミー実装では常に成功するので、このテストは実際のLLM実装時に更新が必要 
