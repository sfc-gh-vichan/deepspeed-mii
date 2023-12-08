import time
import argparse
import mii
from prompt_generator import PromptsGenerator
from threading import Thread

MAX_SEQUENCE_LENGTH = 4096

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument("-k",
                        "--max_new_tokens",
                        type=int,
                        default=1024)
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


class CallbackObject:
    def __init__(self):
        self.responses = []
        self.first = True
        self.first_token_time = 0.0


if __name__ == "__main__":
    client = None
    f = None
    try:
        args = parse_args()
        client = mii.serve(
            model_name_or_path=args.model,
            deployment_name="llama",
            tensor_parallel=args.tensor_parallel,
        )

        callback_object = CallbackObject()
        def callback(response):
            if callback_object.first:
                callback_object.ttft = time.time()
                callback_object.first = False
            callback_object.responses.append(response[0])
            print(response[0])
        
        sampling_params = {
            "max_new_tokens": 50,
            "do_sample": False,
            "top_p": 1.0,
        }

        prompt_generator = PromptsGenerator(tokenizer_path=args.model)
        prompts = []
        for prompt_len in args.prompt_length:
            prompts.append(prompt_generator.generate(
                average_token=prompt_len,
                variance=prompt_len*0.3,
                max_token=MAX_SEQUENCE_LENGTH-args.max_new_tokens,
                n=1,
                show_progress=True,
            ))

        clients = [mii.client("llama") for _ in range(0, len(prompts))]
        threads: list[Thread] = []
        
        def _generate(client, prompt, callback, sampling_params):
            client.generate(prompt, callback, **sampling_params)

        for i, prompt in enumerate(prompts):
            threads.append(Thread(target=_generate, args=[client[i], prompt, callback, sampling_params]))

        start_time = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        end_time = time.time()
        callback_object.ttft = callback_object.ttft - start_time
        latency = end_time - start_time

        # print([out_token.to_msg_dict() for out_token in callback_object.responses])

        # print(' '.join([out_token.generated_text for out_token in callback_object.responses]))
        # print(callback_object.__dict__)
        print("latency: ", latency)
    except Exception as e:
        print(repr(e))
    finally:
        if client is not None:
            client.terminate_server()
        if f is not None:
            f.close()
