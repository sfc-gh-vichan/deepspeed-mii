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

        responses = []
        def callback(response):
            print(response)
            responses.append(response[0])
        
        sampling_params = {
            "max_new_tokens": 50,
            "temperature": 1.0,  
            "do_sample": True,
            "return_full_text": True,
        }

        result_queue = []
        results = client.generate(
            prompts="Hello my name is",
            streaming_fn=callback,
            **sampling_params,
        )

        print([out_token.to_msg_dict() for out_token in responses])

        print(' '.join([out_token.generated_text for out_token in responses]))
    except Exception as e:
        print(repr(e))
    finally:
        if client is not None:
            client.terminate_server()
        if f is not None:
            f.close()
