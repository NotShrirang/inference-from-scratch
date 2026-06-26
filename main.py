from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api import inference_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = {}
    app.state.engine = {}
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(inference_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the ChatModel API!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)