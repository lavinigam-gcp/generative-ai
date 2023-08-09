# Application 1 : Cloud Function that wraps the PaLM Code Bison Models

This application demonstrates a Cloud Function written in Python that initializes the Vertex AI module and then provides an endpoint to invoke PaLM Code Bison model.

## Environment variables required

Your Cloud Function requires access to two environment variables:

- `GCP_PROJECT` : This the Google Cloud Project Id.
- `FUNCTION_REGION` : This is the region in which you are deploying your Cloud Function. For e.g. us-central1.

These variables are needed since the Vertex AI initialization needs the Google Cloud Project Id and the region. The specific code line from the `main.py` function is shown here:
`vertexai.init(project=PROJECT_ID, location=LOCATION)`

These variables can be set via the following [instructions](https://cloud.google.com/functions/docs/configuring/env-var) via any of the following ways:

1. At the time of [deploying](https://cloud.google.com/functions/docs/configuring/env-var#setting_runtime_environment_variables) the Google Cloud Function.
2. [Updating](https://cloud.google.com/functions/docs/configuring/env-var#updating_runtime_environment_variables) the environment variables after deploying the Google Cloud Function.

## Deploying the Cloud Function

Assuming that you have a copy of this project on your local machine with `gcloud` SDK setup on the machine, follow these steps:

1. Go to the root folder of this project.
2. You should have both the `main.py` and `requirements.txt` file present in this folder.
3. Provide the following command. Remember to replace the variables `$PROJECT_ID` and `$REGION` with the correct values for your Google Cloud Project Id and Region name respectively:

   ```bash
   gcloud functions deploy predict \
   --gen2 \
   --runtime=python311 \
   --region=REGION \
   --source=. \
   --entry-point=predictCode \
   --set-env-vars=GCP_PROJECT=$PROJECT_ID,FUNCTION_REGION=$REGION \
   --trigger-http \
   --allow-unauthenticated
   ```

## Invoking the Cloud Function

Since this Cloud Function is deployed with a HTTP trigger, you can directly invoke it. Sample calls are shown below:

```bash
curl -m 70 -X POST https://<REGION>-<GOOGLE_PROJECT_ID>.cloudfunctions.net/predictCode \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Write a Python function to make a call to a URL?"
}'
```
