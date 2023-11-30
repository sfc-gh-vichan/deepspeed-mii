from quart import Quart, request
from relax.flask.handler import Handler
from relax.flask.args import args
from quart import request
from hypercorn.config import Config
import asyncio
from hypercorn.asyncio import serve

from multiprocessing import Process, Lock

mutex = Lock()


config = Config()
config.bind = [f"localhost:{args.port}"]

handler = Handler()

app = Quart(__name__)

@app.post("/generate")
async def _generate():
    data = await request.get_json()
    mutex.acquire()
    model_responses = handler.client.generate(data["prompts"])
    mutex.release()
    return {"model_outputs": [resp.generated_text for resp in model_responses]}


if __name__ == "__main__":
    asyncio.run(serve(app, config))
