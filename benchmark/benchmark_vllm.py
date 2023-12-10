import argparse
import asyncio
import gc
import os
import queue
import random
import time
from typing import List
from transformers import AutoTokenizer
from benchmark_tools import Benchmark, Query, summarize_chat_benchmarks
import threading
import multiprocessing

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


async def benchmark_vllm(
    client,
    prompts: List[str],
    max_new_tokens: int,
    start_time: float,
) -> List[Benchmark]:
    callback_obj = CallbackObject()
    sampling_params = SamplingParams(
        temperature=0,  # get rid of nondeterminism.
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens
    )
    generator = client.generate(prompt=prompts[0], sampling_params=sampling_params, request_id=str(start_time))
    async for request_output in generator:
        outputs = [output for output in request_output.outputs]
        result = outputs[0]
        if callback_obj.first:
            callback_obj.first_token_time = time.time()
            callback_obj.first = False
        callback_obj.responses.append(result)
    end_time = time.time()
    time_to_first_token = callback_obj.first_token_time - start_time
    latency = end_time - start_time

    input_lengths = []
    output_lengths = []

    # input_lengths.append(input_len)
    input_lengths.append(0)
    output_lengths.append(len(callback_obj.responses[-1].token_ids))

    return ([
        Benchmark(
            framework='vllm',
            input_length=input_lengths,
            output_length=output_lengths,
            time_to_first_token=time_to_first_token,
            latency=latency,
            tensor_parallel=8,
        )
    ])
    

def _run_vllm_parallel(
    client,
    barrier,
    query_queue,
    result_queue,
    max_new_tokens,
    client_num,
):
    pid = os.getpid()
    session_id = f"test_session_p{pid}_t{threading.get_ident()}"
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    barrier.wait()

    # Warmup
    try:
        while True:
            query = query_queue.get(timeout=1)
            print(f"warmup queue size: {query_queue.qsize()})", flush=True)
            event_loop.run_until_complete(benchmark_vllm(client=client, prompts=[query.prompt], max_new_tokens=max_new_tokens, start_time=query.start_time))
    except queue.Empty:
        pass

    print(f"Worker ({pid}) finished warmup. session_id: {session_id}")
    barrier.wait()

    time.sleep(random.uniform(0, client_num) * 0.01)
    while True:
        try:
            query = query_queue.get(timeout=30)
            print(f"queue size: {query_queue.qsize()})", flush=True)
            if len(query.prompt) == 0:
                break
            benchmarks = event_loop.run_until_complete(benchmark_vllm(client=client, prompts=[query.prompt], max_new_tokens=max_new_tokens, start_time=query.start_time))
            [result_queue.put(benchmark) for benchmark in benchmarks]
        except queue.Empty:
            pass

    print(f"Worker ({pid}) finished. session_id: {session_id}")


def run_vllm_benchmarks(
    client_num: str,
    use_thread: bool,
    model: str,
    queries_per_second: float,
    prompt_length: int,
    max_new_tokens: int,
    warmup: int,
) -> List[Benchmark]:
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

        # Start threads/processes for # of clients
        if use_thread:
            runnable_cls = threading.Thread
            barrier_cls = threading.Barrier
            queue_cls = queue.Queue
        else:
            runnable_cls = multiprocessing.Process
            barrier_cls = multiprocessing.Barrier
            queue_cls = multiprocessing.Queue
        
        barrier = barrier_cls(client_num + 1)
        query_queue = queue_cls()
        result_queue = queue_cls()

        processes = []
        for _ in range(client_num):
            processes.append(
                runnable_cls(
                    target=_run_vllm_parallel,
                    args=(client, barrier, query_queue, result_queue, max_new_tokens, client_num)
                )
            )
        for p in processes:
            p.start()

        prompt_generator = PromptsGenerator(tokenizer_path=model)

        # Generate warmup prompts. This will generate n * len(prompt_lengths) warmup queries
        prompts = (
            prompt_generator.generate(
                average_token=prompt_length,
                variance=prompt_length*0.3,
                max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                n=warmup*client_num,
                show_progress=True,
            )
        )
        [query_queue.put(Query(prompt)) for prompt in prompts]

        # Barrier to wait for all clients to initialized
        barrier.wait()
        # Barrier for all clients to finish warmup
        barrier.wait()

        time.sleep(5)

        total_queries_sent = 0

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

        # For 30 seconds, send a query every 1/qps
        i = 0
        time_start = time.time()
        while time.time() - time_start < 30:
            if i >= len(prompts):
                i = 0
            query_queue.put(Query(prompts[i]))
            i += 1
            total_queries_sent += 1
            time.sleep(1/queries_per_second)

        response_details = []
        while len(response_details) < total_queries_sent:
            res = result_queue.get(block=True)
            response_details.append(res)

        return response_details
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

    benchmarks = run_vllm_benchmarks(
        client_num=args.client_num,
        use_thread=args.use_thread,
        model=args.model,
        queries_per_second=args.queries_per_second,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
    )

    summarize_chat_benchmarks(
        token_input=args.prompt_length,
        queries_per_second=args.queries_per_second,
        clients=args.client_num,
        benchmarks=sorted(benchmarks),
    )
