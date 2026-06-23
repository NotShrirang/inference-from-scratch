import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
import traceback
import torch

from app.model import ChatModel
from app import logger
from app.api.utils import clear_cuda_cache, get_device


router = APIRouter()

@router.post("/load")
async def load_model(request: Request, model_name: str):
    try:
        model = ChatModel(model_name, device=get_device())
        if request.app.state.model is None:
            request.app.state.model = {}
            logger.info(f"Initialized model storage for the first time.")
        request.app.state.model[model_name] = model
        logger.info(f"Model '{model_name}' loaded successfully.")
        return JSONResponse(content={"message": f"Model '{model_name}' loaded successfully."})
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error loading model '{model_name}': {trace}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unload")
async def unload_model(request: Request, model_name: str):
    clear_cuda_cache()
    if request.app.state.model is None or model_name not in request.app.state.model:
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
async def generate_text(request: Request, model_name: str, input_text: str, max_length: int = 50):
    try:
        model: ChatModel = request.app.state.model.get(model_name)
        logger.info(f"Generating text with model '{model_name}' for input: '{input_text}'")
        if not model:
            raise HTTPException(status_code=400, detail="Model not found.")
        generated_text = model.generate(input_text, max_length)
        return JSONResponse(content={"generated_text": generated_text})
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error generating text with model '{model_name}': {trace}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-stream")
async def generate_text_stream(request: Request, model_name: str, input_text: str, max_length: int = 50):
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
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
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
async def generate_text_batch(request: Request, model_name: str, input_texts: list[str], max_length: int = 50):
    try:
        model: ChatModel = request.app.state.model.get(model_name)
        logger.info(f"Batch text generation with model '{model_name}' for inputs: {input_texts}")
        if not model:
            raise HTTPException(status_code=400, detail="Model not found.")
        generated_texts = model.generate_batch(input_texts, max_length)
        return JSONResponse(content={"generated_texts": generated_texts})
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error in batch text generation with model '{model_name}': {trace}")
        raise HTTPException(status_code=500, detail=str(e))
