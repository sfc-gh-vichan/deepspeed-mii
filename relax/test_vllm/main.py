import argparse
import asyncio
import gc
import torch
import time

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser = AsyncEngineArgs.add_cli_args(parser)
args = parser.parse_args()
engine_args = AsyncEngineArgs.from_cli_args(args)
engine = AsyncLLMEngine.from_engine_args(engine_args)


# get rid of nondeterminism.
sampling_params = SamplingParams(
    temperature=0,  
    top_p=1.0,
    top_k=-1,
    max_tokens=1024
)

results_generator = engine.generate("asdf", sampling_params, "")


async def stream_results():
    full_output = ""
    first = True
    start = time.time()
    ttft = None
    total_time = None
    async for request_output in results_generator:
        if first:
            ttft = time.time() - start
            first = False
        output = ""
        text_outputs = [output.text for output in request_output.outputs]
        for texts in text_outputs:
            output += texts[len(full_output):]
        full_output += output
        yield output
    total_time = time.time() - start
    print("ttft: " + "{:.4f}".format(ttft))
    print("total_time: " + "{:.2f}".format(total_time))


async def main():
    async for result in stream_results():
        print(result)

asyncio.run(main())

try:
    # Destroy
    destroy_model_parallel()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
except Exception as e:
    print(f'failed to destroy vllm: {e}')
