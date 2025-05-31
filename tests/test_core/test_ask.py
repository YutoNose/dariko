from unittest.mock import patch

from dariko import ask, ask_batch, set_config
from tests.conftest import Person, mock_gpt_response


@patch("dariko.models.gpt.requests.post", side_effect=mock_gpt_response)
def test_ask_with_variable_annotation(mock_post):
    """変数アノテーションを使用したaskのテスト"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    result: Person = ask("test", output_model=Person)
    assert isinstance(result, Person)
    assert result.dummy is True


@patch("dariko.models.gpt.requests.post", side_effect=mock_gpt_response)
def test_ask_with_return_type(mock_post):
    """戻り値の型アノテーションを使用したaskのテスト"""
    set_config(model="gpt-4o-mini", llm_key="test_key")

    def get_person(prompt: str) -> Person:
        return ask(prompt, output_model=Person)

    result = get_person("test")
    assert isinstance(result, Person)
    assert result.dummy is True


@patch("dariko.models.gpt.requests.post", side_effect=mock_gpt_response)
def test_ask_with_explicit_model(mock_post):
    """明示的なモデル指定を使用したaskのテスト"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    result = ask("test", output_model=Person)
    assert isinstance(result, Person)
    assert result.dummy is True


@patch("dariko.models.gpt.requests.post", side_effect=mock_gpt_response)
def test_ask_batch(mock_post):
    """バッチ処理のテスト"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    prompts = ["test1", "test2"]
    results: list[Person] = ask_batch(prompts, output_model=Person)
    assert len(results) == 2
    assert all(isinstance(r, Person) for r in results)
    assert all(r.dummy is True for r in results) 
