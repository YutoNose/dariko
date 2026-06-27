from unittest.mock import patch

from dariko import ask, set_config
from tests.conftest import Person


def _mock_list_response(*args, **kwargs):
    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '[{"name": "a", "age": 1, "dummy": false},'
                                ' {"name": "b", "age": 2, "dummy": true}]'
                            )
                        }
                    }
                ]
            }

    return MockResponse()


@patch("dariko.models.gpt.requests.post", side_effect=_mock_list_response)
def test_ask_list_model(mock_post):
    """output_model=list[Person] でリストとして検証される。"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    result = ask("test", output_model=list[Person])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, Person) for r in result)
    assert result[0].name == "a"
    assert result[1].dummy is True
