import functions_framework
from google.cloud import storage
import os

import vertexai
from vertexai.language_models import TextGenerationModel
import google.cloud.logging


PROJECT_ID  = os.environ.get('GCP_PROJECT','-')
LOCATION = os.environ.get('FUNCTION_REGION','-')

client = google.cloud.logging.Client(project=PROJECT_ID)
client.setup_logging()

log_name = "summarize-cloudfunction-log"
logger = client.logger(log_name)

def predictText(prompt, max_output_tokens, temperature, top_p, top_k):
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = TextGenerationModel.from_pretrained("text-bison@001")
    parameters = {
            "temperature": 0.2,
            "max_output_tokens": 256,
            "top_p": 0.8,
            "top_k": 40
        }
    prompt_response = model.predict(prompt,**parameters)

    return prompt_response.text

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def summarize_gcs_object(cloud_event):
    data = cloud_event.data

    bucketname = data["bucket"]
    name = data["name"]

    # Read the contents of the blob
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucketname)
    blob = bucket.blob(name)

    file_contents = blob.download_as_text(encoding="utf-8")

    # Invoke the predict function with the Summarize prompt
    prompt = "Summarize the following: {}".format(file_contents)
    prompt_response = predictText(prompt,1024,0.2,0.8,38)

    # Save the summary in another blob in the summary bucket
    split_names = name.split(".")
    summary_blob_name = "{}-summary.{}".format(split_names[0],split_names[1])
    summarization_bucket = storage_client.bucket("{}-summaries".format(bucketname))
    summary_blob = summarization_bucket.blob(summary_blob_name)
    summary_blob.upload_from_string(prompt_response.encode('utf-8'))
    logger.log("Summarization saved in {}/{}".format(summarization_bucket))