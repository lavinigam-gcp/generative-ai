# Cloud Run application utilizing Streamlit Framework that demonstrates working with Vertex AI API

This application demonstrates a Cloud Run application that uses the [Streamlit](https://streamlit.io/) framework. This project is inspired/taken from the following repository: [https://github.com/lavinigam-gcp/vertex_palm_streamlit_flask/tree/main/demo/streamlit/hello-world-text](https://github.com/lavinigam-gcp/vertex_palm_streamlit_flask/tree/main/demo/streamlit/hello-world-text).

<img src="streamlitapp-screen.png"/>

## Deploying the Application to Cloud Run

To deploy the Streamlit Application in [Cloud Run](https://cloud.google.com/run/docs/quickstarts/deploy-container), you need to build the Docker image in Artifact Registry and deploy it in Cloud Run.

First step is to add your Google Project ID in the `src/vertex.py` file.

Next, look at the following script, replace the variables at the start and run the commands one after the other. This assumes that you have `gcloud` setup on your machine.

```sh
PROJECT_ID=<REPLACE_WITH_YOUR_PROJECT_ID>
REGION=<REPLACE_WITH_YOUR_GCP_REGION_NAME>
AR_REPO=<REPLACE_WITH_YOUR_AR_REPO_NAME>
SERVICE_NAME=chat-streamlit-app
gcloud artifacts repositories create $AR_REPO --location=$REGION --repository-format=Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME
gcloud run deploy $SERVICE_NAME --port 8080 --image $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME --allow-unauthenticated --region=$REGION --platform=managed  --project=$PROJECT_ID
```

