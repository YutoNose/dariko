from unittest.mock import patch

from dariko import ask, set_config
from tests.conftest import Person, mock_gemma_response


@patch("dariko.models.gemma.Gemma.call", side_effect=mock_gemma_response)
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_configure_gemma(mock_model, mock_tokenizer, mock_call):
    """Gemmaモデルの設定テスト"""
    # Hugging Faceのトークンを設定
    set_config(model="google/gemma-2b", llm_key="test_hf_token")
    result: Person = ask("test", output_model=Person)
    assert result.dummy is True 
