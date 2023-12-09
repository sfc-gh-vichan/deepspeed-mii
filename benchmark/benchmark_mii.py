import argparse
import asyncio
import multiprocessing
import os
import queue
import random
import threading
import time
from typing import List
from transformers import AutoTokenizer
from benchmark_tools import Benchmark, summarize_benchmarks

import mii

from prompt_generator import PromptsGenerator
from common_arg_types import list_of_ints

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
        self.responses = []
        self.first = True
        self.first_token_time = 0.0


class Query:
    def __init__(self, prompt):
        self.prompt = prompt
        self.start_time = time.time()


def benchmark_mii(
    client,
    prompts: List[str],
    max_new_tokens: int,
    start_time: float,
) -> List[Benchmark]:
    benchmarks = []
    callback_obj = CallbackObject()

    def callback(response):
        if callback_obj.first:
            callback_obj.first_token_time = time.time()
            callback_obj.first = False
        callback_obj.responses.append(response[0])

    client.generate(
        prompts=prompts,
        streaming_fn=callback,
        do_sample=False,
        top_p=1.0,
        max_new_tokens=max_new_tokens
    )
    end_time = time.time()
    time_to_first_token = callback_obj.first_token_time - start_time
    latency = end_time - start_time

    input_lengths = []
    output_lengths = []

    input_lengths.append(callback_obj.responses[-1].prompt_length)
    output_lengths.append(callback_obj.responses[-1].generated_length)

    benchmarks.append(
        Benchmark(
            framework='mii',
            input_length=input_lengths,
            output_length=output_lengths,
            time_to_first_token=time_to_first_token,
            latency=latency,
            tensor_parallel=8,
        )
    )
    return benchmarks


def _run_mii_parallel(
    model,
    barrier,
    query_queue,
    result_queue,
    max_new_tokens,
    client_num,
) -> None:
    pid = os.getpid()
    session_id = f"test_session_p{pid}_t{threading.get_ident()}"
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    client = mii.client(model)

    barrier.wait()

    # Warmup
    try:
        while True:
            query = query_queue.get(timeout=1)
            print(f"warmup queue size: {query_queue.qsize()} ({pid})", flush=True)
            benchmark_mii(client=client, prompts=[query.prompt], max_new_tokens=max_new_tokens, start_time=query.start_time)
    except queue.Empty:
        pass

    print(f"Worker ({pid}) finished warmup. session_id: {session_id}")
    barrier.wait()

    time.sleep(random.uniform(0, client_num) * 0.01)
    while True:
        try:
            query = query_queue.get(timeout=30) # Get input tokens here as well?
            print(f"queue size: {query_queue.qsize()} ({pid})", flush=True)
            if len(query.prompt) == 0:
                break
            benchmarks = benchmark_mii(client=client, prompts=[query.prompt], max_new_tokens=max_new_tokens, start_time=query.start_time)
            [result_queue.put(benchmark) for benchmark in benchmarks]
        except queue.Empty:
            pass

    print(f"Worker ({pid}) finished. session_id: {session_id}")


def run_mii_benchmarks(
    client_num: int,
    use_thread: bool,
    model: str,
    tensor_parallel: int,
    queries_per_second: float,
    prompt_length: int,
    max_new_tokens: int,
    warmup: int,
) -> List[Benchmark]:
    try:
        # Start mii server
        start = time.time()
        mii.serve(
            model_name_or_path=model,
            deployment_name=model,
            tensor_parallel=tensor_parallel,
            replica_num=1,
        )
        print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start mii engine")

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
                    target=_run_mii_parallel,
                    args=(model, barrier, query_queue, result_queue, max_new_tokens, client_num)
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

        # Tokenizers must be initialized after fork.
        # So we need to fork before putting inputs to the queue.
        # We need this barrier to stop child processse from taking inputs before the main process puts them
        barrier.wait()
        # This barrier is to make sure that all clients have finished warmup
        barrier.wait()

        time.sleep(5)

        total_queries_sent = 0

        # Generate prompts to run benchmark on
        prompts = (
            prompt_generator.generate(
                average_token=prompt_length,
                variance=prompt_length*0.3,
                max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                n=20,
                show_progress=True,
            )
        )
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
            mii.client(model).terminate_server()
        except Exception as e:
            print(f'failed to destroy mii: {e}')


if __name__ ==  "__main__":
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    benchmarks = run_mii_benchmarks(
        client_num=args.client_num,
        use_thread=args.use_thread,
        model=args.model,
        tensor_parallel=args.tensor_parallel,
        queries_per_second=args.queries_per_second,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
    )

    benchmarks = sorted(benchmarks)

    print('!!!---Printing results---!!!')
    # Output results as a csv
    print('framework, input, output, time_to_first_token, latency(s), throughput, tensor_parallel')
    for i in benchmarks:
        print(i)
    
    summarize_benchmarks(
        token_input=args.prompt_length,
        queries_per_second=args.queries_per_second,
        clients=args.client_num,
        benchmarks=benchmarks
    )
