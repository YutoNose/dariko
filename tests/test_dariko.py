import os
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from dariko.config import configure
from dariko.llm import ValidationError, ask, ask_batch


class Person(BaseModel):
    name: str
    age: int
    dummy: bool
    api_key: str


@pytest.fixture(autouse=True)
def set_api_key():
    os.environ["DARIKO_API_KEY"] = "test_key"
    configure()


def mock_llm_response(*args, **kwargs):
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            current_key = os.getenv("DARIKO_API_KEY", "test_key")
            self._json = {
                "choices": [
                    {"message": {"content": f'{{"name": "test", "age": 20, "dummy": true, "api_key": "{current_key}"}}'}}
                ]
            }

        def json(self):
            return self._json

    return MockResponse()


@patch("requests.post", side_effect=mock_llm_response)
def test_configure(mock_post):
    # 環境変数から設定
    os.environ["DARIKO_API_KEY"] = "test_key"
    configure()
    result: Person = ask("test", model_api_import_name="openai", output_model=Person)
    assert result.dummy is True
    assert result.api_key == "test_key"

    # 環境変数を変更
    os.environ["DARIKO_API_KEY"] = "direct_key"
    configure()
    result: Person = ask("test", model_api_import_name="openai", output_model=Person)
    assert result.api_key == "direct_key"


@patch("requests.post", side_effect=mock_llm_response)
def test_ask_with_variable_annotation(mock_post):
    result: Person = ask("test", model_api_import_name="openai", output_model=Person)
    assert isinstance(result, Person)
    assert result.dummy is True


@patch("requests.post", side_effect=mock_llm_response)
def test_ask_with_return_type(mock_post):
    def get_person(prompt: str) -> Person:
        return ask(prompt, model_api_import_name="openai", output_model=Person)

    result = get_person("test")
    assert isinstance(result, Person)
    assert result.dummy is True


@patch("requests.post", side_effect=mock_llm_response)
def test_ask_with_explicit_model(mock_post):
    result = ask("test", model_api_import_name="openai", output_model=Person)
    assert isinstance(result, Person)
    assert result.dummy is True


@patch("requests.post", side_effect=mock_llm_response)
def test_ask_batch(mock_post):
    prompts = ["test1", "test2"]
    results: list[Person] = ask_batch(prompts, model_api_import_name="openai", output_model=Person)
    assert len(results) == 2
    assert all(isinstance(r, Person) for r in results)
    assert all(r.dummy is True for r in results)


def test_validation_error():
    def mock_invalid_response(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.status_code = 200
                self._json = {"choices": [{"message": {"content": '{"invalid": "response"}'}}]}

            def json(self):
                return self._json

        return MockResponse()

    with patch("requests.post", side_effect=mock_invalid_response):
        with pytest.raises(ValidationError):
            ask("invalid", model_api_import_name="openai", output_model=Person)


def mock_gemini_response(*args, **kwargs):
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            current_key = os.getenv("DARIKO_API_KEY", "test_key")
            self._json = {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": f'{{"name": "test", "age": 20, "dummy": true, "api_key": "{current_key}"}}'}
                            ]
                        }
                    }
                ]
            }

        def json(self):
            return self._json

    return MockResponse()


@patch("requests.post", side_effect=mock_gemini_response)
def test_ask_with_gemini(mock_post):
    result: Person = ask("test", model_api_import_name="gemini", output_model=Person)
    assert isinstance(result, Person)
    assert result.dummy is True
    assert result.api_key == "test_key"


@patch("requests.post", side_effect=mock_gemini_response)
def test_ask_batch_with_gemini(mock_post):
    prompts = ["test1", "test2"]
    results: list[Person] = ask_batch(prompts, model_api_import_name="gemini", output_model=Person)
    assert len(results) == 2
    assert all(isinstance(r, Person) for r in results)
    assert all(r.dummy is True for r in results)


def test_invalid_model_api():
    with pytest.raises(RuntimeError, match="モデルのapi呼び出しもとがなにかを確認してください"):
        ask("test", model_api_import_name="invalid_model", output_model=Person)
