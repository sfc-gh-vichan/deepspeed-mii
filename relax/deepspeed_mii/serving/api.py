from typing import Any
from fastapi import FastAPI
from relax.deepspeed_mii.serving.handler import Handler
from relax.deepspeed_mii.serving.schema import ModelInferRequest
import nest_asyncio
nest_asyncio.apply()

handler = Handler()
app = FastAPI()

@app.post("/")
async def serve(req: ModelInferRequest) -> Any:
    model_responses = handler.client.generate(**(req.model_dump()))
    return {"model_outputs": [resp.generated_text for resp in model_responses]}
