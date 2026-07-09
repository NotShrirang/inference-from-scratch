import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
import traceback
import torch

from app.model import ChatModel
from app import logger
from app.request import InferenceRequest
from app.api.utils import clear_cuda_cache, get_device, sample_next_token
from app.batching.engine import ContinuousBatchingEngine

router = APIRouter()

@router.post("/load")
async def load_model(request: Request, batch_size: int = 16, model_name: str = "Qwen/Qwen2.5-0.5B", max_length: int = 50, profiling: bool = False):
    try:
        model = ChatModel(model_name, device=get_device())
        if request.app.state.model is None:
            request.app.state.model = {}
            logger.info(f"Initialized model storage for the first time.")

        if model_name in request.app.state.model.keys():
            logger.info(f"Model '{model_name}' is already loaded. Returning existing model.")
            return JSONResponse(content={
                "message": f"Model '{model_name}' is already loaded.",
                "model_eos_token": model.eos_token,
                "model_pad_token": model.pad_token,
                "model_device": str(model.device),
                "batch_size": batch_size,
                "max_length": max_length
            })

        request.app.state.model[model_name] = model
        logger.info(f"Model '{model_name}' loaded successfully.")
        logger.info(f"Initializing ContinuousBatchingEngine for model '{model_name}'.")
        engine = ContinuousBatchingEngine(model=model, batch_size=batch_size, max_length=max_length, profiling=profiling)
        logger.info(f"Starting continuous batching engine for model '{model_name}'.")
        engine.start()
        request.app.state.engine[model_name] = engine
        return JSONResponse(content={
            "message": f"Model '{model_name}' loaded successfully. Continuous batching engine started.",
            "engine_id": engine.id,
            "model_eos_token": model.eos_token,
            "model_pad_token": model.pad_token,
            "model_device": str(model.device),
            "batch_size": engine.batch_size,
            "max_length": engine.max_length
        })
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error loading model '{model_name}': {trace}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unload")
async def unload_model(request: Request, model_name: str):
    clear_cuda_cache()
    if request.app.state.model is None or model_name not in request.app.state.model.keys():
        raise HTTPException(status_code=400, detail="Model not found.")
    if request.app.state.model is not None:
        logger.info(f"Unloading model '{model_name}'.")
        del request.app.state.model[model_name]
    return JSONResponse(content={"message": f"Model '{model_name}' unloaded successfully."})

@router.get("/models")
async def list_models(request: Request):
    if request.app.state.model is None:
        return JSONResponse(content={"models": []})
    return JSONResponse(content={"models": list(request.app.state.model.keys())})

@router.post("/generate")
async def generate_text(request: Request, model_name: str, input_text: str, max_length: int = 50, temperature: float = 0.6, top_p: float = 0.9):
    try:
        model: ChatModel = request.app.state.model.get(model_name)
        logger.info(f"Generating text with model '{model_name}' for input: '{input_text}'")
        if not model:
            raise HTTPException(status_code=400, detail="Model not found.")
        inputs = model.tokenizer(input_text, return_tensors="pt").to(model.device)
        generated_text = model.generate(inputs, max_length, temperature, top_p)
        return JSONResponse(content={"generated_text": generated_text})
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error generating text with model '{model_name}': {trace}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-stream")
async def generate_text_stream(request: Request, model_name: str, input_text: str, max_length: int = 50, temperature: float = 0.6, top_p: float = 0.9):
    try:
        model: ChatModel = request.app.state.model.get(model_name)
        logger.info(f"Streaming text generation with model '{model_name}' for input: '{input_text}'")
        if not model:
            raise HTTPException(status_code=400, detail="Model not found.")

        async def stream_generator():
            generated_text = input_text
            inputs = model.tokenizer(generated_text, return_tensors="pt")
            past_key_values = None
            for _ in range(max_length):
                inputs = inputs.to(model.device)
                logits, past_key_values = model.forward(inputs, past_key_values=past_key_values)
                next_token_id = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p).item()
                inputs["input_ids"] = torch.tensor([[next_token_id]], device=model.device)
                inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.tensor([[1]], device=model.device)], dim=-1)
                next_token = model.tokenizer.decode(next_token_id)
                generated_text += next_token
                yield f"data: {next_token}\n\n"
                await asyncio.sleep(0)

        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error streaming text generation with model '{model_name}': {trace}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-batch")
async def generate_text_batch(request: Request, model_name: str, input_texts: list[str], max_length: int = 50, temperature: float = 0.6, top_p: float = 0.9):
    try:
        model: ChatModel = request.app.state.model.get(model_name)
        logger.info(f"Batch text generation with model '{model_name}' for inputs: {input_texts}")
        if not model:
            raise HTTPException(status_code=400, detail="Model not found.")
        generated_texts = model.generate_batch(input_texts, max_length, temperature, top_p)
        return JSONResponse(content={"generated_texts": generated_texts})
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error in batch text generation with model '{model_name}': {trace}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-continuous-batch")
async def generate_text_continuous_batch(request: Request, model_name: str = "Qwen/Qwen2.5-0.5B", input_text: str = "Hi", max_length: int = 50, temperature: float = 0.6, top_p: float = 0.9):
    try:
        engine: ContinuousBatchingEngine = request.app.state.engine.get(model_name)
        logger.info(f"Continuous batch text generation with model '{model_name}' for input: {input_text}")
        if not engine:
            raise HTTPException(status_code=400, detail="Continuous batching engine not found for the specified model.")

        req = await engine.submit(prompt=input_text, max_new_tokens=max_length, temperature=temperature, top_p=top_p)

        async def stream_generator():
            while not req.finished:
                if not req.queue.empty():
                    token = await req.queue.get()
                    if token is None:
                        break
                    yield f"data: {engine.model.tokenizer.decode(token, skip_special_tokens=True)}\n\n"
                else:
                    await asyncio.sleep(0.1)
            if req.status == "error":
                raise RuntimeError(getattr(req, "error", "generation failed"))
        
        return StreamingResponse(stream_generator(), media_type="text/plain")
    except RuntimeError as e:
        logger.error(f"Error submitting request to continuous batching engine for model '{model_name}': {str(e)}")
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error in continuous batch text generation with model '{model_name}': {trace}")
        return JSONResponse(status_code=500, content={"detail": str(e), "trace": trace})


@router.get("/engine-profile")
async def engine_profile(request: Request, model_name: str = "Qwen/Qwen2.5-0.5B", reset: bool = False):
    engine: ContinuousBatchingEngine = request.app.state.engine.get(model_name)
    if not engine:
        raise HTTPException(status_code=404, detail="Continuous batching engine not found for the specified model.")
    report = engine.profile_report()
    if reset:
        engine.reset_timings()
    return JSONResponse(content={"engine_id": engine.id, "profile": report})
