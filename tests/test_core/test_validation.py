from unittest.mock import patch

import pytest

from dariko import ask, set_config, ValidationError
from tests.conftest import Person, mock_invalid_response


def test_validation_error():
    """バリデーションエラーのテスト"""
    set_config(model="gpt-4o-mini", llm_key="test_key")

    with patch("dariko.models.gpt.requests.post", side_effect=mock_invalid_response):
        with pytest.raises(ValidationError):
            ask("invalid", output_model=Person)


def test_unsupported_model():
    """未サポートモデルのテスト"""
    with pytest.raises(ValueError, match="Unsupported model"):
        set_config(model="unsupported-model", llm_key="test_key")
        ask("test", output_model=Person) 
