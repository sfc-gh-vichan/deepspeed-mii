import argparse
import asyncio
import gc
import time
from functools import total_ordering
from typing import List
from transformers import AutoTokenizer
import nest_asyncio
nest_asyncio.apply()

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
                        default=128)
    parser.add_argument("-l",
                        "--prompt_length",
                        help="average number of tokens each prompt.",
                        type=list_of_ints,
                        default='1024')
    parser.add_argument('--framework',
                        required=True,
                        type=list_of_strings,
                        default='vllm,mii')
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
        self.responses = []
        self.first = True
        self.first_token_time = 0.0


def benchmark_mii(model: str, tensor_parallel: int, warmup: int, prompt_lengths: List[int], max_new_tokens: int):
    import mii

    start = time.time()
    llm = mii.serve(
        model_name_or_path=model,
        deployment_name=model,
        tensor_parallel=tensor_parallel,
        replica_num=1,
    )
    print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start mii engine")

    prompt_generator = PromptsGenerator(tokenizer_path=model)
    if warmup > 0:
        print('warming up...')
        for prompt_length in prompt_lengths:
            warmup_prompts = prompt_generator.generate(
                average_token=prompt_length,
                variance=max_new_tokens*0.3,
                max_token=max_new_tokens,
                n=warmup
            )
            llm.generate(warmup_prompts, max_new_tokens=max_new_tokens)
        print('warm up finished')

    benchmarks = []
    for prompt_length in prompt_lengths:
        prompt_generator.reset()
        prompts = prompt_generator.generate(
            average_token=prompt_length,
            variance=prompt_length*0.3,
            max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
            n=1,
            show_progress=True
        )

        callback_obj = CallbackObject()
        def callback(response):
            if callback_obj.first:
                callback_obj.first_token_time = time.time()
                callback_obj.first = False
            callback_obj.responses.append(response[0])
            print(response[0])

        start = time.time()
        llm.generate(
            prompts=prompts,
            streaming_fn=callback,
            do_sample=False,
            top_p=1.0,
            max_new_tokens=max_new_tokens
        )
        end = time.time()
        time_to_first_token = callback_obj.first_token_time - start
        latency = end - start
        print("done")

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
                tensor_parallel=tensor_parallel,
            )
        )

        for i in benchmarks:
            print(i)

    try:
        # Destroy
        llm.terminate_server()
    except Exception as e:
        print(f'failed to destroy mii: {e}')
    return benchmarks


async def benchmark_vllm(model: str, tensor_parallel: int, warmup: int, prompt_lengths: List[int], max_new_tokens: int):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

    # Create an LLM.
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args, _ = parser.parse_known_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    start = time.time()
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    print('took ' + "{:.2f}".format(time.time()-start) + " seconds to start vllm engine")

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,  # get rid of nondeterminism.
        top_p=1.0,
        top_k=-1,
        max_tokens=max_new_tokens
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_generator = PromptsGenerator(tokenizer_path=model)
    if warmup > 0:
        print('warming up...')
        for prompt_length in prompt_lengths:
            warmup_prompts = prompt_generator.generate(
                average_token=prompt_length,
                variance=prompt_length*0.3,
                max_token=max_new_tokens,
                n=warmup
            )
            for warmup_prompt in warmup_prompts:
                async for result in llm.generate(warmup_prompt, sampling_params, ""):
                    pass
        print('warm up finished')
    
    async def stream_results(generator):
        async for request_output in generator:
            outputs = [output for output in request_output.outputs]
            yield outputs[0]

    benchmarks = []
    for prompt_length in prompt_lengths:
        prompt_generator.reset()
        prompts = prompt_generator.generate(
            average_token=prompt_length,
            variance=prompt_length*0.3,
            max_token=MAX_SEQUENCE_LENGTH-max_new_tokens,
            n=1,
            show_progress=True
        )

        callback_obj = CallbackObject()
        input_len = len(tokenizer.encode(prompts[0]))

        start = time.time()
        outputs = llm.generate(prompts[0], sampling_params, "")

        async for result in stream_results(outputs):
            if callback_obj.first:
                callback_obj.first_token_time = time.time()
                callback_obj.first = False
            callback_obj.responses.append(result)
            print(result)

        end = time.time()
        time_to_first_token = callback_obj.first_token_time - start
        latency = end - start
        print("done")

        input_lengths = []
        output_lengths = []

        input_lengths.append(input_len)
        output_lengths.append(len(callback_obj.responses[-1].token_ids))

        benchmarks.append(
            Benchmark(
                framework='vllm',
                input_length=input_lengths,
                output_length=output_lengths,
                time_to_first_token=time_to_first_token,
                latency=latency,
                tensor_parallel=tensor_parallel,
            )
        )
        for i in benchmarks:
            print(i)

    try:
        # Destroy
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
    except Exception as e:
        print(f'failed to destroy vllm: {e}')
    return benchmarks


if __name__ == "__main__":
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    benchmarks = []

    if 'mii' in args.framework:
        benchmarks += benchmark_mii(
                model=args.model,
                tensor_parallel=args.tensor_parallel,
                warmup=args.warmup,
                prompt_lengths=args.prompt_length,
                max_new_tokens=args.max_new_tokens)

    if 'vllm' in args.framework:
        benchmarks += asyncio.run(benchmark_vllm(
                model=args.model,
                tensor_parallel=args.tensor_parallel,
                warmup=args.warmup,
                prompt_lengths=args.prompt_length,
                max_new_tokens=args.max_new_tokens))

    benchmarks = sorted(benchmarks)

    print('!!!---Printing results---!!!')
    # Output results as a csv
    print('framework, avg_input, min_input, max_input, avg_output, min_output, max_output, time_to_first_token, latency(s), throughput, tensor_parallel')
    for i in benchmarks:
        print(i)
