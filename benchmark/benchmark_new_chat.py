import argparse
import asyncio
import gc
import multiprocessing
import os
import queue
import random
import threading
import time
from functools import total_ordering
from typing import List
from transformers import AutoTokenizer

import mii
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import torch

from prompt_generator import PromptsGenerator

MAX_SEQUENCE_LENGTH = 4096


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def list_of_strings(arg):
    return arg.split(',')


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
                        type=list_of_ints,
                        default='1024')
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
    parser.add_argument('--model', type=str, required=True, help="path to the model")

    args, _ = parser.parse_known_args()
    return args


@total_ordering
class Benchmark:
    def __init__(self, framework, input_length, output_length, time_to_first_token, latency, tensor_parallel):

        def _avg(lt):
            return sum(lt) // len(lt)

        self.avg_input = _avg(input_length)

        self.framework = framework

        self.max_input = max(input_length)
        self.min_input = min(input_length)

        self.avg_output = _avg(output_length)
        self.max_output = max(output_length)
        self.min_output = min(output_length)

        self.tensor_parallel = tensor_parallel
        self.throughput = (sum(input_length)+sum(output_length))/latency
        self.latency = latency
        self.time_to_first_token = time_to_first_token

    def __str__(self):
        return f'{self.framework}' \
            f', {self.avg_input}, {self.min_input}, {self.max_input}' \
            f', {self.avg_output}, {self.min_output}, {self.max_output}' \
            f', {self.time_to_first_token: .3f}' \
            f', {self.latency: .2f}' \
            f', {self.throughput: .2f}' \
            f', {self.tensor_parallel}'

    def __lt__(self, other):
        if self.avg_input != other.avg_input:
            return self.avg_input < other.avg_input
        if self.tensor_parallel != other.tensor_parallel:
            return self.tensor_parallel < other.tensor_parallel
        if self.framework != other.framework:
            return self.framework < other.framework


class CallbackObject:
    def __init__(self):
        self.responses = []
        self.first = True
        self.first_token_time = 0.0


async def benchmark_mii(
    client,
    prompts: List[str],
    max_new_tokens: int
) -> List[Benchmark]:

    benchmarks = []

    callback_obj = CallbackObject()
    def callback(response):
        if callback_obj.first:
            callback_obj.first_token_time = time.time()
            callback_obj.first = False
        callback_obj.responses.append(response[0])

    start = time.time()
    client.generate(
        prompts=prompts,
        streaming_fn=callback,
        do_sample=False,
        top_p=1.0,
        max_new_tokens=max_new_tokens
    )
    end = time.time()
    time_to_first_token = callback_obj.first_token_time - start
    latency = end - start

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


async def benchmark_vllm(
    client: AsyncLLMEngine,
    prompts: str,
    max_new_tokens: int,
    request_id: str,
) -> List[Benchmark]:
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,  # get rid of nondeterminism.
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens
    )
    
    async def stream_results(generator):
        async for request_output in generator:
            outputs = [output for output in request_output.outputs]
            yield outputs[0]

    benchmarks = []

    callback_obj = CallbackObject()

    start = time.time()
    print("sending inference request on vllm")
    if not client.is_running:
        client.start_background_loop()
    outputs = client.generate(prompts[0], sampling_params, request_id)
    print(client.is_running)
    print("sent inference request on vllm")

    async for result in stream_results(outputs):
        print(outputs)
        if callback_obj.first:
            callback_obj.first_token_time = time.time()
            callback_obj.first = False
        callback_obj.responses.append(result)

    end = time.time()
    time_to_first_token = callback_obj.first_token_time - start
    latency = end - start
    print("finished inference request on vllm")

    input_lengths = []
    output_lengths = []

    # input_lengths.append(input_len)
    input_lengths.append(0)
    output_lengths.append(len(callback_obj.responses[-1].token_ids))

    benchmarks.append(
        Benchmark(
            framework='vllm',
            input_length=input_lengths,
            output_length=output_lengths,
            time_to_first_token=time_to_first_token,
            latency=latency,
            tensor_parallel=8,
        )
    )

    return benchmarks


async def _run_parallel(
    client,
    model,
    num_warmup_queries,
    barrier,
    query_queue,
    result_queue,
    max_new_tokens,
    client_num,
    vllm
) -> None:
    pid = os.getpid()
    session_id = f"test_session_p{pid}_t{threading.get_ident()}"

    # Deepspeed only
    if client is None:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        client = mii.client(model)

    barrier.wait()

    for i in range(num_warmup_queries):
        print(f"warmup queue size: {query_queue.qsize()} ({pid})", flush=True)
        input_prompt = query_queue.get(timeout=1.0)

        if vllm:
            await benchmark_vllm(client, [input_prompt], max_new_tokens, str(i))
        else:
            await benchmark_mii(client, [input_prompt], max_new_tokens)

    barrier.wait()

    time.sleep(random.uniform(0, client_num) * 0.01)
    try:
        while True:
            print(f"queue size: {query_queue.qsize()} ({pid})", flush=True)
            input_prompt = query_queue.get(timeout=1.0) # Get input tokens here as well?
            if len(input_prompt) == 0:
                break

            # Set max_new_tokens following normal distribution
            if vllm:
                benchmarks = await benchmark_vllm(client, [input_prompt], max_new_tokens, str(num_warmup_queries))
            else:
                benchmarks = await benchmark_mii(client, [input_prompt], max_new_tokens)

            [result_queue.put(benchmark) for benchmark in benchmarks]
    except queue.Empty:
        print(f"queue is empty ({pid})")

    print(f"Worker ({pid}) finished. session_id: {session_id}")


async def run_benchmarks(
    client_num: int,
    use_thread: bool,
    model: str,
    tensor_parallel: int,
    prompt_lengths: int,
    max_new_tokens: int,
    warmup: int,
    vllm: bool
) -> List[Benchmark]:
    client = None
    try:
        if vllm:
            parser = argparse.ArgumentParser()
            parser.add_argument("--host", type=str, default="0.0.0.0")
            parser.add_argument("--port", type=int, default=8000)
            parser = AsyncEngineArgs.add_cli_args(parser)
            args, _ = parser.parse_known_args()
            engine_args = AsyncEngineArgs.from_cli_args(args)
            start = time.time()
            client = AsyncLLMEngine.from_engine_args(engine_args)
            print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start vllm engine")
        else:
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

        num_warmup_queries = warmup * len(prompt_lengths)

        processes = []
        for _ in range(client_num):
            processes.append(
                runnable_cls(
                    target=_run_parallel,
                    args=(client, model, num_warmup_queries, barrier, query_queue, result_queue, max_new_tokens, client_num, vllm)
                )
            )
        for p in processes:
            p.start()
        
        prompt_generator = PromptsGenerator(tokenizer_path=model)

        # Generate warmup prompts. This will generate n * len(prompt_lengths) warmup queries
        for prompt_length in prompt_lengths:
            prompts = (
                prompt_generator.generate(
                    average_token=prompt_length,
                    variance=prompt_length*0.3,
                    max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                    n=warmup,
                    show_progress=True,
                )
            )
            [query_queue.put(prompt) for prompt in prompts]
        
        # Generate prompts to run benchmark on
        for prompt_length in prompt_lengths:
            prompts = (
                prompt_generator.generate(
                    average_token=prompt_length,
                    variance=prompt_length*0.3,
                    max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                    n=1,
                    show_progress=True,
                )
            )
            [query_queue.put(prompt) for prompt in prompts]
        
        query_queue.put("")

        # Tokenizers must be initialized after fork.
        # So we need to fork before putting inputs to the queue.
        # We need this barrier to stop child processse from taking inputs before the main process puts them
        barrier.wait()
        # This barrier is to make sure that all clients have finished warmup
        barrier.wait()

        response_details = []
        while len(response_details) < len(prompt_lengths):
            res = result_queue.get()
            # vLLM returns concatinated tokens
            # if vllm:
            #     tokenizer = AutoTokenizer.from_pretrained(model)
            #     all_tokens = tokenizer.tokenize(res.generated_tokens)
            #     res.generated_tokens = all_tokens[len(tokenizer.tokenize(res.prompt)):]
            response_details.append(res)

        return response_details
    except Exception as e:
        print(f"error: {repr(e)}")
    finally:
        if vllm:
            try:
                # Destroy
                destroy_model_parallel()
                del client
                gc.collect()
                torch.cuda.empty_cache()
                torch.distributed.destroy_process_group()
            except Exception as e:
                print(f'failed to destroy vllm: {e}')
        else:
            try:
                # Destroy
                mii.client(model).terminate_server()
            except Exception as e:
                print(f'failed to destroy mii: {e}')


async def main():
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    # benchmarks = run_benchmarks(
    #     client_num=args.client_num,
    #     use_thread=args.use_thread,
    #     model=args.model,
    #     tensor_parallel=args.tensor_parallel,
    #     prompt_lengths=args.prompt_length,
    #     max_new_tokens=args.max_new_tokens,
    #     warmup=args.warmup,
    #     vllm=False,
    # )

    benchmarks = await run_benchmarks(
        client_num=args.client_num,
        use_thread=args.use_thread,
        model=args.model,
        tensor_parallel=args.tensor_parallel,
        prompt_lengths=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        vllm=True,
    )

    benchmarks = sorted(benchmarks)

    print('!!!---Printing results---!!!')
    # Output results as a csv
    print('framework, avg_input, min_input, max_input, avg_output, min_output, max_output, time_to_first_token, latency(s), throughput, tensor_parallel')
    for i in benchmarks:
        print(i)


if __name__ == "__main__":
    asyncio.run(main())