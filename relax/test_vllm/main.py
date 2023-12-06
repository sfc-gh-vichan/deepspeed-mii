import argparse
import asyncio
import gc
import torch
import time

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser = AsyncEngineArgs.add_cli_args(parser)
args = parser.parse_args()
engine_args = AsyncEngineArgs.from_cli_args(args)
engine = AsyncLLMEngine.from_engine_args(engine_args)


# get rid of nondeterminism.
sampling_params = SamplingParams(
    temperature=1.0,  
    top_p=1.0,
    top_k=-1,
    max_tokens=50,
)

tokenizer = AutoTokenizer.from_pretrained(args.model)
results_generator = engine.generate("Hello my name is", sampling_params, "")
input_ids = tokenizer.encode("Hello my name is")
print(len(input_ids))


async def stream_results():
    full_output = ""
    async for request_output in results_generator:
        outputs = [output for output in request_output.outputs]
        # for texts in text_outputs:
        #     output += texts[len(full_output):]
        yield outputs[0]


async def main():
    
    first = True
    ttft = None
    total_time = None
    full_output = ""
    start = time.time()
    last_output_input_ids = 0
    async for result in stream_results():
        last_output_input_ids = len(result.token_ids)
        # if first:
        #     ttft = time.time() - start
        #     first = False
        full_output += result.text[len(full_output):]
    total_time = time.time() - start
    # print("ttft: " + "{:.3f}".format(ttft))
    print("total_time: " + "{:.2f}".format(total_time))
    print(full_output)
    print(len(input_ids))
    print(last_output_input_ids - len(input_ids))


test = asyncio.run(main())

try:
    # Destroy
    destroy_model_parallel()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
except Exception as e:
    print(f'failed to destroy vllm: {e}')
