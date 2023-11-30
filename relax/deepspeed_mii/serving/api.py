from typing import Any
from fastapi import FastAPI
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
    model_responses = handler.client.generate(**(req.model_dump()))
    return {"model_outputs": [resp.generated_text for resp in model_responses]}


@app.post("/generate")
async def generate(request: ModelInferRequest) -> Any:

    channel = grpc.aio.insecure_channel("localhost:50050")
    stub = ModelResponseStub(channel)
    requestData = single_string_request_to_proto(self=None, request_dict=request.model_dump(), **(request.model_dump()))

    # Non-streaming case
    responseData = await stub.GeneratorReply(requestData)
    result = {"text": responseData.response[0]}
    return JSONResponse(result)
