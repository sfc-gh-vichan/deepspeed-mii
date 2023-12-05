import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import Thread
from relax.test_deepspeed.args import args

from dacite import from_dict
import mii

@dataclass
class DeploymentConfig:
    model_name: str
    tensor_parallel: int
    replica_num: int


async def callback(response, queue):
    await queue.put(response)


async def process(result_queue):
    result = await result_queue.get()
    print(result)

async def main():
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
    result_queue = asyncio.Queue()
    results = client.generate(
        prompts="asdf",
        streaming_fn=lambda resp: asyncio.create_task(callback(resp, result_queue)),
    )
    t = Thread(target=process, args=[result_queue])
    t.run()
    await results
    client.terminate_server()
    f.close()


if __name__ == "__main__":
    asyncio.run(main())
