import asyncio
from unittest.mock import patch

from dariko import aask, aask_batch, set_config
from tests.conftest import Person, mock_gpt_response


@patch("dariko.models.gpt.requests.post", side_effect=mock_gpt_response)
def test_aask(mock_post):
    """aask が検証済みオブジェクトを返す。"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    result = asyncio.run(aask("test", output_model=Person))
    assert isinstance(result, Person)
    assert result.dummy is True


@patch("dariko.models.gpt.requests.post", side_effect=mock_gpt_response)
def test_aask_batch_concurrent(mock_post):
    """aask_batch が複数プロンプトを処理する。"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    prompts = ["a", "b", "c"]
    results = asyncio.run(aask_batch(prompts, output_model=Person, concurrency=2))
    assert len(results) == 3
    assert all(isinstance(r, Person) and r.dummy is True for r in results)
