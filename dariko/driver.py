from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as _PydanticValidationError

from .config import get_config, get_llm_key, get_model
from .exceptions import ValidationError
from .model_utils import get_pydantic_model, infer_output_model
from .models.llm import LLM


# モデル名のプレフィックス -> 対応する LLM クラスを返す関数のマッピング。
# Gemma (torch/transformers) を import するのは実際に使う時だけにするため遅延ロードする。
def _gpt_class() -> Type[LLM]:
    from .models.gpt import GPT

    return GPT


def _claude_class() -> Type[LLM]:
    from .models.claude import Claude

    return Claude


def _gemma_class() -> Type[LLM]:
    from .models.gemma import Gemma

    return Gemma


MODEL_MAPPING: Dict[str, Any] = {
    "gpt": _gpt_class,
    "claude": _claude_class,
    "gemma": _gemma_class,
}

# ─────────────────────────────────────────────────────────────
# 内部ユーティリティ
# ─────────────────────────────────────────────────────────────


def _resolve_model(output_model: Type[Any] | None) -> Type[BaseModel]:
    """
    output_model が None の場合は呼び出しフレームから推論し、
    最終的に Pydantic Model 型を返す。
    """
    if output_model is None:
        caller_frame = inspect.currentframe().f_back
        model = infer_output_model(caller_frame)
        if model is None:
            raise TypeError("型アノテーションが取得できませんでした。output_model を指定してください。")
    else:
        model = output_model
    return get_pydantic_model(model)  # 型チェックも兼ねる


def _get_llm_instance() -> LLM:
    """
    設定に基づいて適切な LLM インスタンスを生成する。
    """
    cfg = get_config()
    model_name = get_model()
    llm_key = get_llm_key()

    for prefix, llm_class_factory in MODEL_MAPPING.items():
        if prefix in model_name.lower():
            return llm_class_factory().configure(
                model_name=model_name,
                llm_key=llm_key,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                timeout=cfg.timeout,
            )

    raise ValueError(f"Unsupported model: {model_name}")


def _parse_and_validate(raw_json: str, pyd_model: Type[BaseModel]) -> BaseModel:
    """
    LLM 出力(JSON文字列)を parse & Pydantic 検証。
    成功すれば Pydantic モデルのインスタンスを返す。
    """
    try:
        data = json.loads(raw_json)
        return TypeAdapter(pyd_model).validate_python(data)
    except json.JSONDecodeError as e:
        raise ValidationError(
            _PydanticValidationError.from_exception_data(
                "JSONDecodeError",
                [{"loc": (), "msg": f"LLMの出力がJSONとして解析できませんでした: {e}", "type": "value_error"}],
            )
        ) from None
    except _PydanticValidationError as e:
        raise ValidationError(e) from None


def _run(pyd_model: Type[BaseModel], prompt: str) -> Any:
    """
    1 プロンプトを実行する。検証に失敗した場合はエラー内容を添えて
    最大 ``max_retries`` 回まで LLM に再生成を促す (自己修復)。
    """
    cfg = get_config()
    llm = _get_llm_instance()
    schema = pyd_model.model_json_schema()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": f"次の JSON Schema に厳密に従い、JSON のみを返してください:\n{schema}"},
        {"role": "user", "content": prompt},
    ]

    last_error: ValidationError | None = None
    for _ in range(cfg.max_retries + 1):
        raw = llm.call(messages, response_schema=schema)
        try:
            return _parse_and_validate(raw, pyd_model)
        except ValidationError as e:
            last_error = e
            # 直前の出力とエラーを会話に追加し、修正版を要求する
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"前回の出力は検証に失敗しました: {e}\n"
                        "スキーマに厳密に従い、JSON のみを返してください。"
                    ),
                },
            ]

    assert last_error is not None
    raise last_error


# ─────────────────────────────────────────────────────────────
# パブリック API
# ─────────────────────────────────────────────────────────────
def ask(prompt: str, *, output_model: Type[Any] | None = None) -> Any:
    """
    単一プロンプトを実行し、Pydantic 検証済みオブジェクトを返す。

    検証に失敗した場合は ``set_config(max_retries=...)`` の回数だけ
    LLM へ再生成を促してから ``ValidationError`` を送出する。
    """
    pyd_model = _resolve_model(output_model)
    return _run(pyd_model, prompt)


def ask_batch(prompts: List[str], *, output_model: Type[Any] | None = None) -> List[Any]:
    """
    複数プロンプトをバッチ処理し、検証済みオブジェクトをリストで返す。
    """
    pyd_model = _resolve_model(output_model)
    return [_run(pyd_model, p) for p in prompts]


# ─────────────────────────────────────────────────────────────
# 非同期 API
# ─────────────────────────────────────────────────────────────
async def aask(prompt: str, *, output_model: Type[Any] | None = None) -> Any:
    """
    :func:`ask` の非同期版。ブロッキング I/O を別スレッドへ退避して実行する。
    """
    pyd_model = _resolve_model(output_model)
    return await asyncio.to_thread(_run, pyd_model, prompt)


async def aask_batch(
    prompts: List[str],
    *,
    output_model: Type[Any] | None = None,
    concurrency: Optional[int] = None,
) -> List[Any]:
    """
    複数プロンプトを並行実行する :func:`ask_batch` の非同期版。

    Args:
        prompts: プロンプト列。
        output_model: 出力 Pydantic モデル (省略時は型推論)。
        concurrency: 同時実行数の上限。``None`` なら全件並行。
    """
    pyd_model = _resolve_model(output_model)
    semaphore = asyncio.Semaphore(concurrency) if concurrency else None

    async def _one(p: str) -> Any:
        if semaphore is None:
            return await asyncio.to_thread(_run, pyd_model, p)
        async with semaphore:
            return await asyncio.to_thread(_run, pyd_model, p)

    return await asyncio.gather(*(_one(p) for p in prompts))


# ─────────────────────────────────────────────────────────────
# ストリーミング API
# ─────────────────────────────────────────────────────────────
class StreamedResponse:
    """ストリーミング応答。テキスト増分を逐次 yield しつつ、完了後に検証済みモデルを返す。

    使い方::

        stream = ask_stream("...", output_model=Person)
        for chunk in stream:      # 生成テキストを逐次受け取る
            print(chunk, end="")
        person = stream.result()  # 完了後に Pydantic 検証済みオブジェクト
    """

    def __init__(self, chunks: Iterator[str], pyd_model: Type[BaseModel]):
        self._chunks = chunks
        self._pyd_model = pyd_model
        self._buffer: List[str] = []
        self._result: Any = None
        self._done = False

    def __iter__(self) -> Iterator[str]:
        for chunk in self._chunks:
            self._buffer.append(chunk)
            yield chunk
        self._result = _parse_and_validate("".join(self._buffer), self._pyd_model)
        self._done = True

    @property
    def text(self) -> str:
        """これまでに受信したテキスト全体。"""
        return "".join(self._buffer)

    def result(self) -> Any:
        """検証済みオブジェクトを返す (ストリームを最後まで消費後に呼ぶこと)。"""
        if not self._done:
            raise RuntimeError("ストリームを最後まで消費してから result() を呼んでください")
        return self._result


def ask_stream(prompt: str, *, output_model: Type[Any] | None = None) -> StreamedResponse:
    """
    プロンプトをストリーミング実行し、:class:`StreamedResponse` を返す。

    増分テキストを逐次受け取りつつ、完了後に ``result()`` で検証済みモデルを得る。
    ストリーミングでは自己修復リトライは行わない (検証は完了時に1回)。
    """
    pyd_model = _resolve_model(output_model)
    llm = _get_llm_instance()
    schema = pyd_model.model_json_schema()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": f"次の JSON Schema に厳密に従い、JSON のみを返してください:\n{schema}"},
        {"role": "user", "content": prompt},
    ]
    chunks = llm.call_stream(messages, response_schema=schema)
    return StreamedResponse(chunks, pyd_model)
