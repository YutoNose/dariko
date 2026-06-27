# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 非同期 API: `aask` / `aask_batch` (`concurrency` で同時実行数を制御)
- ストリーミング API: `ask_stream` と `StreamedResponse` (増分テキスト + 完了後に検証済みモデル)
- 非同期・ストリーミングのテスト

## [3.0.0]

### Added
- 検証失敗時の自己修復リトライ (エラー内容を添えて LLM に再生成を促す `max_retries`)
- 構造化出力の強制: OpenAI Structured Outputs (`json_schema`) と Claude tool-use
- `set_config` で `max_tokens` / `temperature` / `timeout` / `max_retries` を設定可能
- 自己修復リトライのテスト

### Changed
- **BREAKING**: `torch` / `transformers` を任意依存に分離。Gemma を使う場合は `pip install "dariko[gemma]"`
- `python-dotenv` を必須依存に追加 (従来は未宣言の暗黙依存だった)
- Claude API 呼び出しにタイムアウトを追加 (従来は無制限)
- `__version__` を `importlib.metadata` から取得 (ハードコードの不整合を解消)

### Fixed
- `import dariko` するだけで torch のロードが必須だった問題を解消 (GPT/Claude 利用時は不要)
