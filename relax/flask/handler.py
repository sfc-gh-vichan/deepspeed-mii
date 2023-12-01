from dataclasses import dataclass
import json
import os
from pathlib import Path
from relax.flask.args import args
from threading import Thread, Lock

from dacite import from_dict
import mii


@dataclass
class DeploymentConfig:
    model_name: str
    tensor_parallel: int
    replica_num: int


class Handler:
    def __init__(self) -> None:
        self.mutex = Lock()
        path = args.model_repository
        p = Path(path)
        deployment_config = self._load_deployment_config(p=p)
        self.client = mii.serve(
            model_name_or_path=os.path.join(path, self.model_name),
            deployment_name=deployment_config.model_name,
            tensor_parallel=deployment_config.tensor_parallel,
            replica_num=deployment_config.replica_num,
        )

    def _load_deployment_config(self, p: Path) -> DeploymentConfig:
        for f in p.iterdir():
            if f.is_dir():
                self.model_name = f.name
                break
        if len(self.model_name) == 0:
            # model does not exist
            pass
        deployment_config_file_path = os.path.join(p, self.model_name, "deployment_config.json")
        with open(deployment_config_file_path) as f:
            config = json.load(f)
            return from_dict(data_class=DeploymentConfig, data=config)