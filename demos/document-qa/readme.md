# Docy - Document Q & A with PaLM API

To setup this application, we will assume that you have provisioned a Google Cloud Project with Billing enabled. Assuming that you are using a local or Cloud Shell environment with gcloud setup, first thing is to setup the Google Project ID and Region variables.

```sh
PROJECT_ID=<REPLACE_WITH_YOUR_PROJECT_ID>
REGION=<REPLACE_WITH_YOUR_GCP_REGION_NAME>
```

## Enable the Google Cloud Services

This application uses the following services in Google Cloud

- Cloud Run
- Artifact Registry
- Cloud Build
- Vertex AI Services
- Cloud Document AI

Ensure that the following services are enabled for your Google Cloud Project:

- Cloud Run Admin API
- Artifact Registry API
- Cloud Logging API
- Cloud Build API
- Vertex AI API
- Document AI API

## Create an Artifact Registry

We will build a containerized version of the Streamlit application and deploy the same on Google Cloud Run. We will be using Cloud Build to run the builds. The Container Images that we generate will be saved in the Artifact Registry of our choice. Go ahead and create the Artifact Registry as per the command given below:

```sh
AR_REPO=<REPLACE_WITH_YOUR_AR_REPO_NAME>
gcloud artifacts repositories create $AR_REPO --location=$REGION --repository-format=Docker
```

## Create a Document AI Processor

1. From Google Cloud Console, go to Document AI --> Processor Gallery.
2. In the General Category, you will see **Document OCR** , click on **Create Processor**
3. Give a Processor Name, let the region remain **US** and click **Create**

Note down the **Name** and **ID** of the Processor. We will refer to them as `DOCAI_PROCESSOR_DISPLAY_NAME` and `DOCAI_PROCESSOR_ID` variables.

## Build and deploy in Cloud Run

To deploy the Streamlit Application in [Cloud Run](https://cloud.google.com/run/docs/quickstarts/deploy-container), you need to build the Docker image in Artifact Registry and deploy it in Cloud Run.

### Update the Application Configuration file

First step is to visit the `src/env.yaml` file and set a few key properties in the file.

- PROJECT_ID : Provide the Google Project ID value here.
- LOCATION_NAME : Provide the Google Project ID location. Go with us-central1 for now.
- DOCAI_PROCESSOR_DISPLAY_NAME : The name of the Document Processor created.
- DOCAI_PROCESSOR_ID : The ID of the Document Processor created.

Next, create a name for the Cloud Run Service (e.g. mydocqaservice)

```sh
SERVICE_NAME=<REPLACE_WITH_YOUR_SERVICE_NAME>
```

### Build the Container Image

```sh
gcloud auth configure-docker $REGION-docker.pkg.dev
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME
```

### Deploy the Cloud Run Service

```sh
gcloud run deploy $SERVICE_NAME --port 8080 --image $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME --allow-unauthenticated --region=$REGION --platform=managed  --project=$PROJECT_ID
```

### Configure Session Affinity for Cloud Run service

The Cloud Run service will need Session Affinity to be setup. Check out [Session Affinity](https://cloud.google.com/run/docs/configuring/session-affinity) document on how to configure this for your Cloud Run service.
