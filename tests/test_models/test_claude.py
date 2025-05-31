from unittest.mock import patch

from dariko import ask, set_config
from tests.conftest import Person, mock_claude_response


@patch("dariko.models.claude.requests.post", side_effect=mock_claude_response)
def test_configure_claude(mock_post):
    """Claudeモデルの設定テスト"""
    # AnthropicのAPIキーを設定
    set_config(model="claude-3-opus-20240229", llm_key="test_anthropic_key")
    result: Person = ask("test", output_model=Person)
    assert result.dummy is True 
