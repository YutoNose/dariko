from unittest.mock import patch

import pytest

from dariko import ask_stream, set_config
from tests.conftest import Person


class _StreamResp:
    """requests.post(..., stream=True) の戻り値を模した SSE レスポンス。"""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self._lines

    def raise_for_status(self):
        pass


def _gpt_sse(parts):
    """content 増分の列を OpenAI SSE 行 (bytes) に変換する。"""
    lines = []
    for p in parts:
        body = '{"choices":[{"delta":{"content":%s}}]}' % _json_str(p)
        lines.append(("data: " + body).encode("utf-8"))
    lines.append(b"data: [DONE]")
    return lines


def _json_str(s):
    import json

    return json.dumps(s)


def test_ask_stream_yields_and_validates():
    """増分テキストを逐次受け取り、完了後に検証済みオブジェクトを返す。"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    parts = ['{"name": "test", ', '"age": 20, ', '"dummy": true}']

    with patch("dariko.models.gpt.requests.post", return_value=_StreamResp(_gpt_sse(parts))):
        stream = ask_stream("test", output_model=Person)
        received = list(stream)  # ストリームを消費
        result = stream.result()

    assert "".join(received) == "".join(parts)
    assert isinstance(result, Person)
    assert result.dummy is True


def test_ask_stream_partials():
    """partials() が部分モデルを progressively yield し、完了後に検証済みモデルを返す。"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    parts = ['{"name": "test"', ', "age": 20', ', "dummy": true}']

    with patch("dariko.models.gpt.requests.post", return_value=_StreamResp(_gpt_sse(parts))):
        stream = ask_stream("test", output_model=Person)
        snapshots = [p.model_dump() for p in stream.partials()]
        result = stream.result()

    # name だけ -> name+age -> 全部、と段階的に埋まる
    assert {"name": "test", "age": None, "dummy": None} in snapshots
    assert {"name": "test", "age": 20, "dummy": None} in snapshots
    assert snapshots[-1] == {"name": "test", "age": 20, "dummy": True}
    assert isinstance(result, Person)
    assert result.dummy is True


def test_result_before_consume_raises():
    """消費前に result() を呼ぶとエラー。"""
    set_config(model="gpt-4o-mini", llm_key="test_key")
    sse = _gpt_sse(['{"name":"x","age":1,"dummy":false}'])
    with patch("dariko.models.gpt.requests.post", return_value=_StreamResp(sse)):
        stream = ask_stream("test", output_model=Person)
        with pytest.raises(RuntimeError):
            stream.result()
