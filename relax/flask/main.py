from flask import Flask, request
from relax.flask.handler import Handler
from relax.flask.args import args
from werkzeug.serving import make_server
from relax.flask.schema import GenerateSchema
from http import HTTPStatus
from marshmallow.exceptions import ValidationError


handler = Handler()

app = Flask(__name__)

app.teardown_appcontext_funcs.append(handler.client.terminate_server())

@app.route('/generate', methods=['POST'])
def _generate():
    schema = GenerateSchema()
    request_data = request.get_json()
    try:
        generate_kwargs = schema.load(request_data)
        print(generate_kwargs)
        model_responses = handler.client.generate(**generate_kwargs)
        return {"model_outputs": [resp.__dict__ for resp in model_responses]}
    except ValidationError as e:
        return {"error": e.messages}, HTTPStatus.BAD_REQUEST
    except Exception as e:
        return {"error": e.messages}, HTTPStatus.INTERNAL_SERVER_ERROR

@app.route('/terminate', methods=['POST'])
def _terminate():
    handler.client.terminate_server()
    return {"response": "ok"}

if __name__ == "__main__":
    try:
        server = make_server(
            host="0.0.0.0",
            port=args.port,
            app=app,
        ).serve_forever()
    except BaseException as e:
        print("terminating server")
        handler.client.terminate_server()
