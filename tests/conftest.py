import os
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from dariko import set_config


class Person(BaseModel):
    name: str
    age: int
    dummy: bool


@pytest.fixture(autouse=True)
def set_api_key():
    """テスト用のAPIキーを設定する"""
    os.environ["DARIKO_API_KEY"] = "test_key"
    set_config(model="gpt-4o-mini", llm_key="test_key")


def mock_gpt_response(*args, **kwargs):
    """GPTモデルのモックレスポンス"""
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self._json = {
                "choices": [
                    {
                        "message": {
                            "content": '{"name": "test", "age": 20, "dummy": true}'
                        }
                    }
                ]
            }

        def json(self):
            return self._json

        def raise_for_status(self):
            pass

    return MockResponse()


def mock_gemma_response(*args, **kwargs):
    """Gemmaモデルのモックレスポンス"""
    return '{"name": "test", "age": 20, "dummy": true}'


def mock_claude_response(*args, **kwargs):
    """Claudeモデルのモックレスポンス"""
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self._json = {
                "content": [
                    {
                        "text": '{"name": "test", "age": 20, "dummy": true}'
                    }
                ]
            }

        def json(self):
            return self._json

        def raise_for_status(self):
            pass

    return MockResponse()


def mock_invalid_response(*args, **kwargs):
    """無効なレスポンスのモック"""
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self._json = {"choices": [{"message": {"content": '{"invalid": "response"}'}}]}

        def json(self):
            return self._json

    return MockResponse() 
