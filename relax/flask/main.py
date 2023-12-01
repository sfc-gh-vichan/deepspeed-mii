from flask import Flask, request
from relax.flask.handler import Handler
from relax.flask.args import args
handler = Handler()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def _generate():
    data = request.get_json()
    handler.mutex.acquire()
    model_responses = handler.client.generate(data["prompts"])
    handler.mutex.release()
    return {"model_outputs": [resp.__dict__ for resp in model_responses]}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
