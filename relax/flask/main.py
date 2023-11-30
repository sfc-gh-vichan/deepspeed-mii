from flask import Flask, request
from relax.flask.handler import Handler
from relax.flask.args import args

handler = Handler()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def _generate():
    model_responses = handler.client.generate(request.prompts)
    return {"model_outputs": [resp.generated_text for resp in model_responses]}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)