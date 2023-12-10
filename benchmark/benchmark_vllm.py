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
from common_arg_types import list_of_floats, list_of_ints

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
                        default=128)
    parser.add_argument("-l",
                        "--prompt_length",
                        help="average number of tokens each prompt.",
                        type=list_of_ints,
                        default="512,1024,1536,2048,2560")
    parser.add_argument("-tp",
                        "--tensor_parallel",
                        type=int,
                        help="Tensor parallelism",
                        default='1')
    parser.add_argument("-c",
                        "--client_num",
                        type=int,
                        help="Number of clients",
                        default=64)
    parser.add_argument("-t",
                        "--use_thread",
                        action="store_true",
                        help="use thread for clients, else multiprocessing",
                        default=False)
    parser.add_argument("-qps",
                        "--queries_per_second",
                        type=list_of_floats,
                        help="List of queries per second",
                        default="0.5,1.0,1.5,2.0")
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
    model: str,
    prompts: List[str],
    max_new_tokens: int,
    query: Query,
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

    def get_streaming_response(response: requests.Response, time_last_token) -> Iterable[List[str]]:
        for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False,
                                        delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"][0]
                time_now = time.time()
                yield output, time_now - time_last_token
                time_last_token = time_now

    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    token_gen_time = []
    last_response = ""
    for h, t in get_streaming_response(response, query.start_time):
        last_response = h
        token_gen_time.append(t)

    time_to_first_token = token_gen_time[0]
    latency = time.time() - query.start_time

    tokenizer = AutoTokenizer.from_pretrained(model)
    output_token_ids = tokenizer.encode(last_response)

    input_length = [query.input_tokens]
    output_length = [len(output_token_ids) - query.input_tokens]

    benchmarks = ([
        Benchmark(
            framework='vllm',
            input_length=input_length,
            output_length=output_length,
            time_to_first_token=time_to_first_token,
            latency=latency,
            tensor_parallel=8,
        )
    ])

    return benchmarks
    

def _run_vllm_parallel(
    model,
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
            benchmark_vllm(model=model, prompts=[query.prompt], max_new_tokens=max_new_tokens, query=query)
    except queue.Empty:
        pass

    print(f"Worker ({pid}) finished warmup. session_id: {session_id}")
    barrier.wait()

    time.sleep(random.uniform(0, client_num) * 0.01)
    while True:
        try:
            query = query_queue.get(timeout=30)
            if len(query.prompt) == 0:
                break
            benchmarks = benchmark_vllm(model=model, prompts=[query.prompt], max_new_tokens=max_new_tokens, query=query)
            [result_queue.put(benchmark) for benchmark in benchmarks]
        except queue.Empty:
            pass

    print(f"Worker ({pid}) finished. session_id: {session_id}")


def run_vllm_benchmarks(
    client_num: int,
    use_thread: bool,
    model: str,
    queries_per_second_list: List[float],
    prompt_length_list: List[int],
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
                    args=(model, barrier, query_queue, result_queue, max_new_tokens, client_num)
                )
            )
        for p in processes:
            p.start()

        prompt_generator = PromptsGenerator(tokenizer_path=model)

        # Generate warmup prompts. This will generate n * len(prompt_lengths) warmup queries
        prompts = (
            prompt_generator.generate(
                average_token=2560,
                variance=2560*0.3,
                max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                n=warmup,
                show_progress=True,
            )
        )
        [query_queue.put(Query(prompt)) for prompt in prompts]

        # Barrier to wait for all clients to initialized
        barrier.wait()
        # Barrier for all clients to finish warmup
        barrier.wait()

        time.sleep(5)

        for prompt_length in prompt_length_list:
            for queries_per_second in queries_per_second_list:
                print(f"benchmarking {prompt_length} prompt length at {queries_per_second} qps")
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

                # For 5 minutes, send a query every 1/qps
                i = 0
                total_queries_sent = 0
                time_start = time.time()
                while time.time() - time_start < 300:
                    if i >= len(prompts):
                        i = 0
                    query_queue.put(Query(prompts[i]))
                    i += 1
                    total_queries_sent += 1
                    time.sleep(1/queries_per_second)

                benchmarks = []
                while len(benchmarks) < total_queries_sent:
                    res = result_queue.get(block=True)
                    benchmarks.append(res)

                summarize_chat_benchmarks(
                    framework="vllm",
                    token_input=prompt_length,
                    queries_per_second=queries_per_second,
                    clients=args.client_num,
                    benchmarks=sorted(benchmarks),
                )
        
        for _ in range(client_num):
            query_queue.put(Query(("", 0)))

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
        queries_per_second_list=args.queries_per_second,
        prompt_length_list=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
    )
