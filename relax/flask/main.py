from flask import Flask, request
from relax.flask.handler import Handler

handler = Handler()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def _generate():
    model_responses = handler.client.generate(request.prompts)
    return {"model_outputs": [resp.generated_text for resp in model_responses]}
