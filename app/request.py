import typing
from datetime import datetime
import asyncio


class InferenceRequest:
    request_id: str
    model_name: str
    max_tokens: int
    prompt: str
    prompt_tokens: list[str]
    prompt_len: int
    temperature: float
    top_p: float
    tokens: list[str]
    kv_cache: typing.Optional[typing.Any]
    queue: asyncio.Queue
    created_at: str
    updated_at: str
    status: str
    finished: bool

    def __init__(self, request_id: str, model_name: str, max_tokens: int, prompt: str, prompt_tokens: list[str], prompt_len: int, tokens: list[str], temperature: float, top_p: float, created_at: str, updated_at: str, status: str, finished: bool):
        self.request_id = str(request_id)
        self.model_name = str(model_name)
        self.max_tokens = max_tokens
        self.prompt = str(prompt)
        self.prompt_tokens = prompt_tokens
        self.prompt_len = prompt_len
        self.prompt_len = 0
        self.tokens = tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kv_cache = None
        self.queue = asyncio.Queue()
        self.created_at = created_at
        self.updated_at = updated_at
        self.status = str(status)
        self.finished = finished

    def update_status(self, status: str) -> None:
        self.status = status
        self.updated_at = datetime.utcnow().isoformat()
    
    def mark_finished(self) -> None:
        self.finished = True
        self.status = "finished"
        self.updated_at = datetime.utcnow().isoformat()

    def add_token(self, token: str) -> None:
        self.tokens.append(token)
        self.updated_at = datetime.utcnow().isoformat()

    def stop(self) -> None:
        self.finished = True
        self.status = "stopped"
        self.queue.put_nowait(None)
        self.updated_at = datetime.utcnow().isoformat()

    def clean(self) -> None:
        self.tokens = []
        self.kv_cache = None
        self.queue = asyncio.Queue()
        self.status = "cleaned"
        self.finished = True
        self.updated_at = datetime.utcnow().isoformat()
