from functools import total_ordering
from typing import List

def avg(lt):
    return sum(lt) // len(lt)

@total_ordering
class Benchmark:
    def __init__(self, framework, input_length, output_length, time_to_first_token, latency, tensor_parallel):

        self.avg_input = avg(input_length)

        self.framework = framework

        self.max_input = max(input_length)
        self.min_input = min(input_length)

        self.avg_output = avg(output_length)
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


def summarize_benchmarks(
    token_input: int,
    queries_per_second: int,
    clients: int,
    benchmarks: List[Benchmark],
) -> None:
    min_token_input = min([benchmark.max_input for benchmark in benchmarks])
    avg_token_input = avg([benchmark.max_input for benchmark in benchmarks])
    max_token_input = max([[benchmark.max_input for benchmark in benchmarks]])

    min_token_output = min([benchmark.max_output for benchmark in benchmarks])
    avg_token_output = min([benchmark.max_output for benchmark in benchmarks])
    max_token_output = min([benchmark.max_output for benchmark in benchmarks])

    min_time_to_first_token = min([benchmark.time_to_first_token for benchmark in benchmarks])
    avg_time_to_first_token = min([benchmark.time_to_first_token for benchmark in benchmarks])
    max_time_to_first_token = min([benchmark.time_to_first_token for benchmark in benchmarks])

    min_latency = min([benchmark.latency for benchmark in benchmarks])
    avg_latency = min([benchmark.latency for benchmark in benchmarks])
    max_latency = min([benchmark.latency for benchmark in benchmarks])

    # Sum up total input and output tokens, divide by the time it took to complete entire benchmark (last_request_end_time - first_request_start_time)
    # avg_throughput = (avg_token_input+avg_token_output)/avg_latency
    
    print(f"token_input: {token_input}")
    print(f"queries_per_second: {queries_per_second}")
    print(f"clients: {clients}")

    print(f"min_token_input: {min_token_input}")
    print(f"avg_token_input: {avg_token_input}")
    print(f"max_token_input: {max_token_input}")

    print(f"min_token_output: {min_token_output}")
    print(f"avg_token_output: {avg_token_output}")
    print(f"max_token_output: {max_token_output}")

    print(f"min_time_to_first_token: {min_time_to_first_token}")
    print(f"avg_time_to_first_token: {avg_time_to_first_token}")
    print(f"max_time_to_first_token: {max_time_to_first_token}")

    print(f"min_latency: {min_latency}")
    print(f"avg_latency: {avg_latency}")
    print(f"max_latency: {max_latency}")
    