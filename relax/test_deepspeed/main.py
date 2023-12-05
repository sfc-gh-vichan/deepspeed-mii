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


async def stream(response):
    return None


if __name__ == "__main__":
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
    results = client.generate(
        prompts="asdf",
        streaming_fn=stream,
    )
    print(results)
    client.terminate_server()
    f.close()
