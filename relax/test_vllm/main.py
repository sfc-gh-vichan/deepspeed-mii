import argparse
from time import sleep
import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams


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

results_generator = engine.generate("asdf", sampling_params, "")


async def stream_results():
    full_output = ""
    prev_len = 0
    async for request_output in results_generator:
        output = ""
        text_outputs = [output.text for output in request_output.outputs]
        for texts in text_outputs:
            output += texts[len(full_output):]
        full_output += output
        yield output

# results = stream_results()

# print(results)

# sleep(1000)

async def main():
    async for result in stream_results():
        print(result)

asyncio.run(main())