import argparse
import asyncio
import gc
import queue
import time
from functools import total_ordering
from typing import List
from transformers import AutoTokenizer

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
        self.start_time = time.time()
        self.responses = []
        self.first = True
        self.first_token_time = 0.0


async def run_vllm_benchmarks(
    model: str,
    prompt_lengths: int,
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
        for prompt_length in prompt_lengths:
            prompts.append(
                prompt_generator.generate(
                    average_token=prompt_length,
                    variance=prompt_length*0.3,
                    max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                    n=warmup,
                    show_progress=True,
                )
            )
        [print(type(prompt)) for _, prompt in enumerate(prompts)]
        [client.generate(prompt=prompt, sampling_params=sampling_params, request_id=str(10000 + i)) for i, prompt in enumerate(prompts)]

        # Prompts for benchmarking
        prompts = []
        for prompt_length in prompt_lengths:
            prompts.append(
                prompt_generator.generate(
                    average_token=prompt_length,
                    variance=prompt_length*0.3,
                    max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
                    n=1,
                    show_progress=True,
                )
            )

        async def stream(generator):
            async for request_output in generator:
                outputs = [output for output in request_output.outputs]
                yield outputs[0]

        async def stream_results(outputs, benchmark_queue: queue.Queue, callback_obj: CallbackObject):
            async for result in stream(outputs):
                print(outputs)
                if callback_obj.first:
                    callback_obj.first_token_time = time.time()
                    callback_obj.first = False
                callback_obj.responses.append(result)
            end_time = time.time()
            time_to_first_token = callback_obj.first_token_time - callback_obj.start_time
            latency = end_time - callback_obj.start_time

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

        tasks = []
        benchmark_queue = queue.Queue()
        for i, prompt in enumerate(prompts):
            print(type(prompt))
            callback_obj = CallbackObject()
            outputs = client.generate(prompt=prompt, sampling_params=sampling_params, request_id=str(i))
            tasks.append(asyncio.create_task(stream_results(outputs, benchmark_queue, callback_obj)))
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
        model=args.model,
        prompt_lengths=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
    ))

    benchmarks = sorted(benchmarks)

    print('!!!---Printing results---!!!')
    # Output results as a csv
    print('framework, avg_input, min_input, max_input, avg_output, min_output, max_output, time_to_first_token, latency(s), throughput, tensor_parallel')
    for i in benchmarks:
        print(i)
