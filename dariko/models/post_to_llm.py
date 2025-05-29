import re
import requests
from ..config import get_api_key, get_model

# URL定数
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def post_to_llm(messages: list[dict[str, str]], model_api_import_name: str) -> str:
    """
    LLM APIを呼び出して content 文字列を返す。
    """
    api_key = get_api_key()
    if model_api_import_name == "openai":
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
    elif model_api_import_name == "gemini":
        # Gemini APIの正しい形式
        model_name = get_model()
        url = _GEMINI_URL.format(model=model_name)
        
        # MessagesをGemini API形式に変換
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                # systemメッセージはuserメッセージとして扱う
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System: {msg['content']}"}]
                })
            else:
                contents.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["content"]}]
                })
        
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
                }
            },
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Gemini API呼び出しに失敗しました: {r.status_code} - {r.text}")
        
        response = r.json()
        if "candidates" not in response or not response["candidates"]:
            raise RuntimeError(f"Gemini APIレスポンスが不正です: {response}")
        
        raw_text = response["candidates"][0]["content"]["parts"][0]["text"]
        
        # Geminiが返すテキストからJSON部分を抽出
        # マークダウンのコードブロックや余分なテキストを除去
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        else:
            # JSONが見つからない場合はそのまま返す
            return raw_text
    else:
        raise RuntimeError(f"モデルのapi呼び出しもとがなにかを確認してください。: {model_api_import_name}")