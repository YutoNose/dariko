from pydantic import ValidationError as _PydanticValidationError


class ValidationError(Exception):
    """LLM 出力の型検証エラーを表す例外"""

    def __init__(self, original: _PydanticValidationError):
        super().__init__(str(original))
        self.original = original 
