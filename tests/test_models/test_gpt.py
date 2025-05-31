import os
from unittest.mock import patch

import pytest

from dariko import ask, set_config
from tests.conftest import Person, mock_gpt_response


@patch("dariko.models.gpt.requests.post", side_effect=mock_gpt_response)
def test_configure_gpt(mock_post):
    """GPTモデルの設定テスト"""
    # 環境変数から設定
    os.environ["DARIKO_API_KEY"] = "test_key"
    set_config(model="gpt-4o-mini", llm_key="test_key")
    result: Person = ask("test", output_model=Person)
    assert result.dummy is True

    # 環境変数を変更
    os.environ["DARIKO_API_KEY"] = "direct_key"
    set_config(model="gpt-4o-mini", llm_key="direct_key")
    result: Person = ask("test", output_model=Person)
    assert result.dummy is True 
