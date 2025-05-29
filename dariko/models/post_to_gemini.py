import re
import requests
from ..config import get_api_key, get_model

# URL定数
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


def post_to_gemini(messages: list[dict[str, str]]) -> str:
    """
    Gemini APIを呼び出して content 文字列を返す。
    """
    model_name = get_model()
    url = _GEMINI_URL.format(model=model_name)
    api_key = get_api_key()

    # MessagesをGemini API形式に変換
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            # systemメッセージはuserメッセージとして扱う
            contents.append(
                {"role": "user", "parts": [{"text": f"System: {msg['content']}"}]}
            )
        else:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})

    r = requests.post(
        f"{url}?key={api_key}",
        headers={
            "Content-Type": "application/json",
        },
        json={
            "contents": contents,
            "generationConfig": {
                "temperature": 0.1,
                "candidateCount": 1,
                "maxOutputTokens": 2048,
            },
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Gemini API呼び出しに失敗しました: {r.status_code} - {r.text}"
        )

    response = r.json()
    if "candidates" not in response or not response["candidates"]:
        raise RuntimeError(f"Gemini APIレスポンスが不正です: {response}")

    raw_text = response["candidates"][0]["content"]["parts"][0]["text"]

    # Geminiが返すテキストからJSON部分を抽出
    # 1. まずマークダウンのコードブロックを確認
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # 2. 完全なJSONオブジェクトを抽出（中括弧のバランスを考慮）
    brace_count = 0
    start_pos = raw_text.find("{")
    if start_pos != -1:
        for i, char in enumerate(raw_text[start_pos:], start_pos):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return raw_text[start_pos : i + 1].strip()

    # 3. JSONが見つからない場合はそのまま返す
    return raw_text.strip()
