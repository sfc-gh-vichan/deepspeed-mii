from dataclasses import dataclass
import json
import os
from pathlib import Path
from relax.deepspeed_mii.args import args

from dacite import from_dict
import mii


@dataclass
class DeploymentConfig:
    model_name: str
    tensor_parallel: int
    replica_num: int


class Handler:
    def __init__(self) -> None:
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
        with open(deployment_config_file_path) as f:
            config = json.load(f)
            deployment_config = from_dict(data_class=DeploymentConfig, data=config)
            mii.serve(
                model_name_or_path=os.path.join(path, model_name),
                deployment_name=deployment_config.model_name,
                tensor_parallel=deployment_config.tensor_parallel,
                replica_num=deployment_config.replica_num,
            )
