import asyncio
import time
import typing
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import torch

from app import logger
from app.api.utils import sample_next_token, sample_next_token_batched
from app.model import ChatModel
from app.request import InferenceRequest

class ContinuousBatchingEngine:
    def __init__(self, model: ChatModel, batch_size: int = 4, max_length: int = 50, profiling: bool = False):
        self.id = f"cbe-{uuid.uuid4()}"
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_waiting_requests = 4 * self.batch_size
        self.waiting_requests: typing.List[InferenceRequest] = []
        self.active_requests: typing.List[InferenceRequest] = []
        self.attn_mask = None
        self.past_key_values = None
        self.dirty = False
        self.running = False

        # profiling: when on, _timeit syncs CUDA + records per-op timings. OFF by default so the
        # serving hot path doesn't pay a torch.cuda.synchronize() on every forward.
        self.profiling = profiling
        self.timings = defaultdict(lambda: [0.0, 0])  # name -> [total_seconds, count]

        self.model.tokenizer.padding_side = "left"
        if self.model.pad_token is None:
            self.model.pad_token = self.model.eos_token

        logger.info(f"ContinuousBatchingEngine for {self.model.model_name} initialized with batch size {batch_size} and max length {max_length}")
    
    @contextmanager
    def _timeit(self, name: str):
        """Time a block, syncing CUDA so async kernels are attributed correctly.
        No-op unless self.profiling — the syncs serialize the GPU and must not tax serving."""
        if not self.profiling:
            yield
            return
        cuda = str(self.model.device).startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            rec = self.timings[name]
            rec[0] += time.perf_counter() - t0
            rec[1] += 1

    def profile_report(self) -> dict:
        return {
            name: {"avg_ms": round(total / count * 1000, 4), "count": count,
                   "total_ms": round(total * 1000, 2)}
            for name, (total, count) in self.timings.items() if count
        }

    def reset_timings(self):
        self.timings.clear()

    def _active_empty(self) -> bool:
        return self.active_requests_count == 0
    
    @property
    def active_requests_count(self) -> int:
        return len(self.active_requests)
    
    @property
    def waiting_requests_count(self) -> int:
        return len(self.waiting_requests)
    
    def _tokenize_prompts(self, prompt: str) -> torch.Tensor:
        return self.model.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)

    def _prefill(self) -> torch.Tensor:
        # logger.info(f"Prefilling for {self.active_requests_count} active requests.")
        # left pad every active reqs full tokens to max length; req.tokens is a list of token ids
        maxlen = max(len(req.tokens) for req in self.active_requests)
        padded_tokens = [
            [self.model.pad_token] * (maxlen - len(req.tokens)) + req.tokens for req in self.active_requests
        ]
        inputs = {
            "input_ids": torch.tensor(padded_tokens, device=self.model.device),
            "attention_mask": torch.tensor([[0] * (maxlen - len(req.tokens)) + [1] * len(req.tokens) for req in self.active_requests], device=self.model.device),
        }

        self.attn_mask = inputs["attention_mask"]

        # logger.info(f"Forward pass for {self.active_requests_count} active requests.")
        with self._timeit("prefill_fwd"):
            logits, past_key_values = self.model.forward(inputs)
        return logits, past_key_values
    
    def _decode(self, past_key_values) -> typing.Tuple[torch.Tensor, typing.Any]:
        # logger.info(f"Decoding for {self.active_requests_count} active requests.")
        # feed the last token and kv cache of each active request to the model
        last_tokens = torch.tensor(
            [[req.tokens[-1]] for req in self.active_requests],
            device=self.model.device,
        )
        inputs = {"input_ids": last_tokens, "attention_mask": self.attn_mask}
        with self._timeit("decode_fwd"):
            logits, past_key_values = self.model.forward(inputs, past_key_values=past_key_values)
        return logits, past_key_values

    async def submit(self, prompt: str, max_new_tokens: int, temperature: float = 0.6, top_p: float = 0.9) -> InferenceRequest:
        if self.waiting_requests_count >= self.max_waiting_requests:
            raise RuntimeError(f"Too many waiting requests ({self.waiting_requests_count}). Please try again later.")
        logger.info(f"Submitting new request with prompt: {prompt[:10]} and max_new_tokens: {max_new_tokens}")
        with self._timeit("tokenize"):
            inputs = self._tokenize_prompts(prompt)
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name=self.model.model_name,
            max_tokens=max_new_tokens,
            prompt=prompt,
            prompt_tokens=inputs["input_ids"].tolist()[0],
            prompt_len=len(inputs["input_ids"].tolist()[0]),
            tokens=inputs["input_ids"].tolist()[0],
            temperature=temperature,
            top_p=top_p,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            status="pending",
            finished=False
        )
        logger.info(f"Request submitted: {request.request_id} with prompt: {prompt[:10]} and max_new_tokens: {max_new_tokens}")
        self.waiting_requests.append(request)
        self.dirty = True
        return request
    
    def start(self):
        if self.running:
            return
        self.running = True
        asyncio.create_task(self._run())

    async def _run(self):
        logger.info(f"ContinuousBatchingEngine with id {self.id} started.")
        while self.running:
            if self.active_requests_count < self.batch_size and self.waiting_requests:
                # Move requests from waiting to active
                while self.active_requests_count < self.batch_size and self.waiting_requests:
                    request = self.waiting_requests.pop(0)
                    request.update_status("active")
                    # logger.info(f"Moving request {request.request_id} to active requests.")
                    self.active_requests.append(request)
                    self.dirty = True

            # If active requests are not there, waiting for some time before checking again
            if self._active_empty():
                await asyncio.sleep(0.005)
                continue
                
            # step (guarded: one bad step must not silently kill the engine loop)
            try:
                self._step()
            except Exception:
                logger.error("engine _step failed; resetting active batch and continuing", exc_info=True)
                self._reset_batch()

            # SSE responses flush
            await asyncio.sleep(0)

    def _step(self):
        if self._active_empty():
            return
        
        # prefill if self.dirty, else decode
        if self.dirty:
            # logger.info(f"Dirty flag is set. Prefilling for active requests.")
            logits, past_key_values = self._prefill()
            self.past_key_values = past_key_values
            self.dirty = False
        else:
            # logger.info(f"Dirty flag is not set. Decoding for active requests.")
            logits, past_key_values = self._decode(self.past_key_values)
            self.past_key_values = past_key_values

        temps = torch.tensor([r.temperature for r in self.active_requests], device=self.model.device).unsqueeze(1)
        tops  = torch.tensor([r.top_p for r in self.active_requests], device=self.model.device).unsqueeze(1)

        with self._timeit("sample"):
            next_ids = sample_next_token_batched(logits[:, -1, :], temperature=temps, top_p=tops).tolist()

        for i, request in enumerate(self.active_requests):
            next_token_id = next_ids[i]
            if (str(next_token_id) == str(self.model.eos_token)) or ((len(request.tokens) - request.prompt_len) >= request.max_tokens):
                # logger.info(f"Request {request.request_id} finished. Marking as finished and removing from active requests.")
                request.mark_finished()
                request.add_token(None)
                request.queue.put_nowait(None)
                self.dirty = True
            else:
                # logger.info(f"Request {request.request_id} generated token id {next_token_id}. Adding to tokens.")
                request.add_token(next_token_id)
                request.queue.put_nowait(next_token_id)

        self.active_requests = [req for req in self.active_requests if not req.finished]

        if not self.dirty and self.attn_mask is not None and self.active_requests_count > 0:
            self.attn_mask = torch.cat([self.attn_mask, torch.ones((self.active_requests_count, 1), dtype=self.attn_mask.dtype, device=self.model.device)], dim=-1)

    def _reset_batch(self):
        """Recover from a failed step: fail the in-flight batch (so clients unblock),
        drop the cache, and free CUDA memory — the engine keeps serving instead of dying."""
        for req in self.active_requests:
            if not req.finished:
                req.mark_error("engine step failed (likely CUDA OOM)")
                req.queue.put_nowait(None)   # sentinel so the client's stream unblocks
        self.active_requests = []
        self.past_key_values = None
        self.attn_mask = None
        self.dirty = False
        if str(self.model.device).startswith("cuda"):
            torch.cuda.empty_cache()

    async def stop(self):
        logger.info(f"Stopping ContinuousBatchingEngine with id {self.id}")
        self.running = False
