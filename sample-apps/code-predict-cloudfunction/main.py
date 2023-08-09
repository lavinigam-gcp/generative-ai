import functions_framework
import os
import json

from google.cloud import aiplatform
import google.cloud.logging

import vertexai
from vertexai.preview.language_models import CodeGenerationModel

PROJECT_ID  = os.environ.get('GCP_PROJECT','-')
LOCATION = os.environ.get('FUNCTION_REGION','-')

client = google.cloud.logging.Client(project=PROJECT_ID)
client.setup_logging()

log_name = "predictCode-cloudfunction-log"
logger = client.logger(log_name)

@functions_framework.http
def predictCode(request):

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'prompt' in request_json:
        prompt = request_json['prompt']
        logger.log("Received request for prompt: {}".format(prompt))
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 1024
        }
        model = CodeGenerationModel.from_pretrained("code-bison@001")
        prompt_response = model.predict(prompt,**parameters)
        logger.log("PaLM Code Bison Model response: {}".format(prompt_response.text))
    else:
        prompt_response = 'No prompt provided.'
    
    return json.dumps({"response_text":prompt_response.text})