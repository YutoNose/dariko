from unittest.mock import patch

import pytest

from dariko import ValidationError, ask, set_config
from tests.conftest import Person


class _Resp:
    def __init__(self, content):
        self.status_code = 200
        self._json = {"choices": [{"message": {"content": content}}]}

    def json(self):
        return self._json


def test_self_healing_retry_recovers():
    """初回は不正JSON、2回目で正しい出力 -> リトライで回復する。"""
    set_config(model="gpt-4o-mini", llm_key="test_key", max_retries=2)

    responses = iter(
        [
            _Resp('{"invalid": "response"}'),  # 検証失敗
            _Resp('{"name": "test", "age": 20, "dummy": true}'),  # 修正版
        ]
    )

    with patch("dariko.models.gpt.requests.post", side_effect=lambda *a, **k: next(responses)):
        result: Person = ask("test", output_model=Person)

    assert result.dummy is True
    assert result.name == "test"


def test_self_healing_retry_exhausted():
    """常に不正な出力なら max_retries 後に ValidationError。"""
    set_config(model="gpt-4o-mini", llm_key="test_key", max_retries=1)

    with patch(
        "dariko.models.gpt.requests.post",
        side_effect=lambda *a, **k: _Resp('{"invalid": "response"}'),
    ):
        with pytest.raises(ValidationError):
            ask("test", output_model=Person)
