from typing import Any
from fastapi import FastAPI
from fastapi.logger import logger
import grpc
from relax.deepspeed_mii.serving.handler import Handler
from relax.deepspeed_mii.serving.schema import ModelInferRequest
from mii.grpc_related.proto.modelresponse_pb2_grpc import ModelResponseStub
from mii.grpc_related.task_methods import single_string_request_to_proto
from fastapi.responses import JSONResponse

handler = Handler()
app = FastAPI()

@app.post("/")
async def serve(req: ModelInferRequest) -> Any:
    logger.info("/")
    model_responses = await handler.client.generate(req.prompts)
    return {"model_outputs": [resp.generated_text for resp in model_responses]}


@app.post("/generate")
async def generate(req: ModelInferRequest) -> Any:
    logger.info("/generate")
    channel = grpc.aio.insecure_channel("localhost:50050")
    stub = ModelResponseStub(channel)
    requestData = single_string_request_to_proto(self=None, request_dict={"query": req.prompts})
    responseData = await stub.GeneratorReply(requestData)
    result = {"text": responseData.response[0]}
    logger.info(result)
    print(result)
    return JSONResponse(result)
