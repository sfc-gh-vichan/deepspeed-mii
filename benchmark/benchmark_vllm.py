import argparse
import asyncio
import gc
import json
import os
import queue
import random
import time
from typing import Iterable, List
import requests
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


def benchmark_vllm(
    prompts: List[str],
    max_new_tokens: int,
    start_time: float,
) -> List[Benchmark]:
    api_url = "http://localhost:8000/generate"
    headers = {"User-Agent": "Benchmark Client"}
    pload = {
        "prompt": prompts[0],
        "n": 1,
        "use_beam_search": False,
        "temperature": 0,
        "top_p": 0.9,
        "top_k": 1,
        "max_tokens": max_new_tokens,
        "ignore_eos": False,
        "stream": True,
    }
    def clear_line(n: int = 1) -> None:
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        for _ in range(n):
            print(LINE_UP, end=LINE_CLEAR, flush=True)

    def get_streaming_response(response: requests.Response, time_last_token) -> Iterable[List[str]]:
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False,
                                        delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"][0]
                time_now = time.time()
                yield output, time_now - time_last_token
                time_last_token = time_now

    def get_response(response: requests.Response) -> List[str]:
        data = json.loads(response.content)
        output = data["text"]
        return output

    start_time = time.time()
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    token_gen_time = []
    for h, t in get_streaming_response(response, start_time):
        output = h
        token_gen_time.append(t)
    
    time_to_first_token = token_gen_time[0]

    benchmarks = ([
        Benchmark(
            framework='mii',
            input_length=[0],
            output_length=[0],
            time_to_first_token=time_to_first_token,
            latency=0,
            tensor_parallel=8,
        )
    ])

    return benchmarks
    

def _run_vllm_parallel(
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
            print(f"warmup queue size: {query_queue.qsize()} ({pid})", flush=True)
            benchmark_vllm(prompts=[query.prompt], max_new_tokens=max_new_tokens, start_time=query.start_time)
    except queue.Empty:
        pass

    print(f"Worker ({pid}) finished warmup. session_id: {session_id}")
    barrier.wait()

    time.sleep(random.uniform(0, client_num) * 0.01)
    while True:
        try:
            query = query_queue.get(timeout=30)
            print(f"warmup queue size: {query_queue.qsize()} ({pid})", flush=True)
            if len(query.prompt) == 0:
                break
            benchmarks = benchmark_vllm(prompts=[query.prompt], max_new_tokens=max_new_tokens, start_time=query.start_time)
            [result_queue.put(benchmark) for benchmark in benchmarks]
        except queue.Empty:
            pass

    print(f"Worker ({pid}) finished. session_id: {session_id}")


def run_vllm_benchmarks(
    client_num: int,
    use_thread: bool,
    model: str,
    queries_per_second: float,
    prompt_length: int,
    max_new_tokens: int,
    warmup: int,
) -> List[Benchmark]:
    try:
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
                    args=(barrier, query_queue, result_queue, max_new_tokens, client_num)
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
