import argparse
import asyncio
import gc
import queue
import time
from typing import List
from transformers import AutoTokenizer
from benchmark_tools import Benchmark, Query, summarize_chat_benchmarks
import threading
import nest_asyncio
nest_asyncio.apply()

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import torch

from prompt_generator import PromptsGenerator

MAX_SEQUENCE_LENGTH = 4096


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument("-k",
                        "--max_new_tokens",
                        type=int,
                        default=1024)
    parser.add_argument("-w",
                        "--warmup",
                        type=int,
                        help="number of queries for warming up",
                        default=3)
    parser.add_argument("-l",
                        "--prompt_length",
                        help="average number of tokens each prompt.",
                        type=int,
                        default=1024)
    parser.add_argument("-tp",
                        "--tensor_parallel",
                        type=int,
                        help="Tensor parallelism",
                        default='1')
    parser.add_argument("-c",
                        "--client_num",
                        type=int,
                        help="Number of clients",
                        default='1')
    parser.add_argument("-t",
                        "--use_thread",
                        action="store_true",
                        help="use thread for clients, else multiprocessing",
                        default=False)
    parser.add_argument("-qps",
                        "--queries_per_second",
                        type=float,
                        help="List of queries per second",
                        default=0.5)
    parser.add_argument('--model', type=str, required=True, help="path to the model")

    args, _ = parser.parse_known_args()
    return args


class CallbackObject:
    def __init__(self):
        self.start_time = time.time()
        self.responses = []
        self.first = True
        self.first_token_time = 0.0


async def run_vllm_benchmarks(
    client_num: str,
    model: str,
    queries_per_second: float,
    prompt_length: int,
    max_new_tokens: int,
    warmup: int,
) -> List[Benchmark]:
    benchmarks: List[Benchmark] = []
    client = None
    try:
        # Start vllm server
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8000)
        parser = AsyncEngineArgs.add_cli_args(parser)
        args, _ = parser.parse_known_args()
        engine_args = AsyncEngineArgs.from_cli_args(args)
        start = time.time()
        client = AsyncLLMEngine.from_engine_args(engine_args)
        print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start vllm engine")

        sampling_params = SamplingParams(
            temperature=0,  # get rid of nondeterminism.
            top_p=1.0,
            top_k=-1,
            max_tokens=max_new_tokens
        )

        prompt_generator = PromptsGenerator(tokenizer_path=model)

        # Warmup
        prompts = []
        prompts = (
            prompt_generator.generate(
                average_token=prompt_length,
                variance=prompt_length*0.3,
                max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                n=warmup*client_num,
                show_progress=True,
            )
        )
        [client.generate(prompt=prompt, sampling_params=sampling_params, request_id=str(10000 + i)) for i, prompt in enumerate(prompts)]

        # Prompts for benchmarking
        # Generate prompts to run benchmark on
        prompts = (
            prompt_generator.generate(
                average_token=prompt_length,
                variance=prompt_length*0.3,
                max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                n=100,
                show_progress=True,
            )
        )

        async def stream(generator):
            async for request_output in generator:
                outputs = [output for output in request_output.outputs]
                yield outputs[0]

        async def stream_results(outputs, benchmark_queue: queue.Queue, query: Query):
            callback_obj = CallbackObject()
            print("==========================================================")
            async for result in stream(outputs):
                if callback_obj.first:
                    callback_obj.first_token_time = time.time()
                    callback_obj.first = False
                callback_obj.responses.append(result)
            end_time = time.time()
            time_to_first_token = callback_obj.first_token_time - query.start_time
            latency = end_time - query.start_time

            input_lengths = []
            output_lengths = []

            # input_lengths.append(input_len)
            input_lengths.append(0)
            output_lengths.append(len(callback_obj.responses[-1].token_ids))

            benchmark_queue.put(
                Benchmark(
                    framework='vllm',
                    input_length=input_lengths,
                    output_length=output_lengths,
                    time_to_first_token=time_to_first_token,
                    latency=latency,
                    tensor_parallel=8,
                )
            )

        # For 30 seconds, send a query every 1/qps
        tasks = []
        benchmark_queue = queue.Queue()
        i = 0
        loop = asyncio.get_event_loop()
        time_start = time.time()
        while time.time() - time_start < 30:
            if i >= len(prompts):
                i = 0
            query = Query(prompts[i])
            print(f"generating query {i}")
            outputs = client.generate(prompt=query.prompt, sampling_params=sampling_params, request_id=str(i))
            tasks = loop.create_task(stream_results(outputs, benchmark_queue, query))
            i += 1
            time.sleep(1/queries_per_second)
        await asyncio.gather(*tasks)
        return list(benchmark_queue.queue)
    except Exception as e:
        print(f"error: {repr(e)}")
    finally:
        try:
            # Destroy
            if client is not None:
                destroy_model_parallel()
                del client
                gc.collect()
                torch.cuda.empty_cache()
                torch.distributed.destroy_process_group()
        except Exception as e:
            print(f'failed to destroy vllm: {e}')


if __name__ ==  "__main__":
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    benchmarks = asyncio.run(run_vllm_benchmarks(
        client_num=args.client_num,
        model=args.model,
        queries_per_second=args.queries_per_second,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
    ))

    summarize_chat_benchmarks(
        token_input=args.prompt_length,
        queries_per_second=args.queries_per_second,
        clients=args.client_num,
        benchmarks=sorted(benchmarks),
    )
