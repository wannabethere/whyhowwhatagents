from typing import Any, Optional


class SwarmException(Exception):
    def __init__(
        self, message: str, status_code: int, detail: Optional[Any] = None
    ):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class SwarmDocumentProcessingError(SwarmException):
    def __init__(self, error_message, document_id):
        self.document_id = document_id
        super().__init__(error_message, 400, {"document_id": document_id})
