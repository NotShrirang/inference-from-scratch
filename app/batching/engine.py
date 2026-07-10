import asyncio
import time
import typing
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import torch
import torch.nn.functional as F

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
        self.attn_mask = None          # invariant: shape[1] == current cache seq length
        self.past_key_values = None
        self.new_requests: typing.List[InferenceRequest] = []   # admitted, not yet prefilled/spliced
        self.dirty = False
        self.running = False

        self.profiling = profiling
        self.timings = defaultdict(lambda: [0.0, 0])  # name -> [total_seconds, count]

        self.model.tokenizer.padding_side = "left"
        if self.model.pad_token is None:
            self.model.pad_token = self.model.eos_token

        logger.info(f"ContinuousBatchingEngine for {self.model.model_name} initialized with batch size {batch_size} and max length {max_length}")
    
    @contextmanager
    def _timeit(self, name: str):
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

    @staticmethod
    def _position_ids(mask: torch.Tensor) -> torch.Tensor:
        return (mask.long().cumsum(-1) - 1).clamp_min_(0)

    def _prefill(self) -> torch.Tensor:
        new = self.new_requests
        maxlen = max(len(req.tokens) for req in new)
        input_ids = torch.tensor(
            [[self.model.pad_token] * (maxlen - len(r.tokens)) + r.tokens for r in new],
            device=self.model.device,
        )
        new_mask = torch.tensor(
            [[0] * (maxlen - len(r.tokens)) + [1] * len(r.tokens) for r in new],
            device=self.model.device,
        )
        with self._timeit("prefill_fwd"):
            logits, new_kv = self.model.forward(
                {"input_ids": input_ids, "attention_mask": new_mask, "position_ids": self._position_ids(new_mask)})
        if self.past_key_values is None:
            self.past_key_values, self.attn_mask = new_kv, new_mask
        else:
            self._splice(new_kv, new_mask)
        return logits

    def _decode(self) -> torch.Tensor:
        last_tokens = torch.tensor([[r.tokens[-1]] for r in self.active_requests], device=self.model.device)
        decode_mask = torch.cat(
            [self.attn_mask, torch.ones((self.active_requests_count, 1), dtype=self.attn_mask.dtype, device=self.model.device)],
            dim=-1)
        pos = self._position_ids(decode_mask)[:, -1:]   # position of just the new token, per row
        with self._timeit("decode_fwd"):
            logits, past_key_values = self.model.forward(
                {"input_ids": last_tokens, "attention_mask": decode_mask, "position_ids": pos},
                past_key_values=self.past_key_values)
        self.past_key_values = past_key_values
        self.attn_mask = decode_mask     # keeps attn_mask.shape[1] == cache seq length
        return logits

    @staticmethod
    def _lpad_seq(t, m):
        return F.pad(t, (0, 0, m - t.shape[2], 0))

    def _splice(self, new_kv, new_mask):
        m = max(self.attn_mask.shape[1], new_mask.shape[1])
        for old, fresh in zip(self.past_key_values.layers, new_kv.layers):
            old.keys = torch.cat([self._lpad_seq(old.keys, m), self._lpad_seq(fresh.keys, m)], dim=0)
            old.values = torch.cat([self._lpad_seq(old.values, m), self._lpad_seq(fresh.values, m)], dim=0)
        self.attn_mask = torch.cat(
            [F.pad(self.attn_mask, (m - self.attn_mask.shape[1], 0)),
             F.pad(new_mask, (m - new_mask.shape[1], 0))], dim=0)

    def _trim_left_pad(self):
        if self.attn_mask is None or self.attn_mask.numel() == 0:
            return
        lead = int(self.attn_mask.argmax(dim=1).min().item())  # first real col per row -> common min
        if lead <= 0:
            return
        self.attn_mask = self.attn_mask[:, lead:]
        for layer in self.past_key_values.layers:
            layer.keys = layer.keys[:, :, lead:, :]
            layer.values = layer.values[:, :, lead:, :]

    def _evict_finished(self):
        keep = [i for i, r in enumerate(self.active_requests) if not r.finished]
        if not keep:
            self.past_key_values, self.attn_mask, self.active_requests = None, None, []
            return
        idx = torch.tensor(keep, device=self.model.device)
        for layer in self.past_key_values.layers:
            layer.keys = layer.keys.index_select(0, idx)
            layer.values = layer.values.index_select(0, idx)
        self.attn_mask = self.attn_mask.index_select(0, idx)
        self.active_requests = [self.active_requests[i] for i in keep]
        self._trim_left_pad()   # reclaim columns the departed (longest) rows were padding for

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
                    self.new_requests.append(request)   # needs prefill + splice on the next step
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

        if self.new_requests:
            batch = self.new_requests
            logits = self._prefill()
            self.new_requests = []
            self.dirty = False
        else:
            batch = self.active_requests
            logits = self._decode()

        temps = torch.tensor([r.temperature for r in batch], device=self.model.device).unsqueeze(1)
        tops  = torch.tensor([r.top_p for r in batch], device=self.model.device).unsqueeze(1)
        with self._timeit("sample"):
            next_ids = sample_next_token_batched(logits[:, -1, :], temperature=temps, top_p=tops).tolist()

        for i, request in enumerate(batch):
            next_token_id = next_ids[i]
            if (str(next_token_id) == str(self.model.eos_token)) or ((len(request.tokens) - request.prompt_len) >= request.max_tokens):
                request.mark_finished()
                request.add_token(None)
                request.queue.put_nowait(None)
            else:
                request.add_token(next_token_id)
                request.queue.put_nowait(next_token_id)

        if any(r.finished for r in self.active_requests):
            self._evict_finished()

    def _reset_batch(self):
        """Recover from a failed step: fail the in-flight batch (so clients unblock),
        drop the cache, and free CUDA memory — the engine keeps serving instead of dying."""
        for req in self.active_requests:
            if not req.finished:
                req.mark_error("engine step failed (likely CUDA OOM)")
                req.queue.put_nowait(None)   # sentinel so the client's stream unblocks
        self.active_requests = []
        self.new_requests = []
        self.past_key_values = None
        self.attn_mask = None
        self.dirty = False
        if str(self.model.device).startswith("cuda"):
            torch.cuda.empty_cache()

    async def stop(self):
        logger.info(f"Stopping ContinuousBatchingEngine with id {self.id}")
        self.running = False
