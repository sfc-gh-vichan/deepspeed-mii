from flask import Flask, request
from relax.flask.handler import Handler
from relax.flask.args import args
from werkzeug.serving import make_server
from relax.flask.schema import GenerateSchema


handler = Handler()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def _generate():
    schema = GenerateSchema()
    request_data = request.get_json()
    result = schema.load(request_data)
    print(result)
    model_responses = handler.client.generate(data["prompts"])
    return {"model_outputs": [resp.__dict__ for resp in model_responses]}


if __name__ == "__main__":
    server = make_server(
        host="0.0.0.0",
        port=args.port,
        app=app,
    ).serve_forever()
