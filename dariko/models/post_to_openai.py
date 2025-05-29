import requests
from ..config import get_api_key, get_model

# URL定数
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def post_to_openai(messages: list[dict[str, str]]) -> str:
    """
    OpenAI APIを呼び出して content 文字列を返す。
    """
    api_key = get_api_key()
    r = requests.post(
        _OPENAI_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": get_model(),
            "messages": messages,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI API呼び出しに失敗しました: {r.text}")
    return r.json()["choices"][0]["message"]["content"]