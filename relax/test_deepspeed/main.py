from dataclasses import dataclass
import json
import os
from pathlib import Path
from relax.test_deepspeed.args import args

from dacite import from_dict
import mii

@dataclass
class DeploymentConfig:
    model_name: str
    tensor_parallel: int
    replica_num: int

if __name__ == "__main__":
    client = None
    f = None
    try:
        path = args.model_repository
        p = Path(path)
        model_name = ""
        for f in p.iterdir():
            if f.is_dir():
                model_name = f.name
                break
        if len(model_name) == 0:
            # model does not exist
            pass
        deployment_config_file_path = os.path.join(path, model_name, "deployment_config.json")
        f = open(deployment_config_file_path)
        config = json.load(f)
        deployment_config = from_dict(data_class=DeploymentConfig, data=config)
        client = mii.serve(
            model_name_or_path=os.path.join(path, model_name),
            deployment_name=deployment_config.model_name,
            tensor_parallel=deployment_config.tensor_parallel,
            replica_num=deployment_config.replica_num,
        )

        out_tokens = []
        def callback(response):
            print(type(response[0]))
            print(f"recv: {response[0]}")
            out_tokens.append(response[0])

        result_queue = []
        results = client.generate(
            prompts="asdf",
            streaming_fn=callback,
            max_new_tokens=50,
        )

        print(' '.join([out_token.generated_text for out_token in out_tokens]))
    except Exception as e:
        print(repr(e))
    finally:
        if client is not None:
            client.terminate_server()
        if f is not None:
            f.close()
