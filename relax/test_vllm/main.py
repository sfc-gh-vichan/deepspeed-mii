import argparse
from time import sleep

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser = AsyncEngineArgs.add_cli_args(parser)
args = parser.parse_args()
engine_args = AsyncEngineArgs.from_cli_args(args)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Requests

sampling_params = SamplingParams(temperature=0,  # get rid of nondeterminism.
    top_p=1.0,
    top_k=-1,
    max_tokens=1024
)

results_generator = engine.generate(["asdf"], sampling_params, "1")

print("generating...")

def stream_results():
    for request_output in results_generator:
        prompt = request_output.prompt
        text_outputs = [
            prompt + output.text for output in request_output.outputs
        ]
        ret = {"text": text_outputs}
        print(ret)
        yield ret

stream_results()


sleep(1000)