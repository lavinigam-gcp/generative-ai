# Cloud Run application utilizing Gradio Framework that demonstrates working with Vertex AI API
This application demonstrates a Cloud Run application that uses the [Gradio](https://www.gradio.app/) framework. 

<img src="images/text-demo.png"/>

## Deploying the Application to Cloud Run

To deploy the Gradio in [Cloud Run](https://cloud.google.com/run/docs/quickstarts/deploy-container), you need to build the Docker image in Artifact Registry and deploy it in Cloud Run.

First step is to add your Google Project ID in the `app.py` file. 

Next, look at the following script, replace the variables at the start and run the commands one after the other. This assumes that you have `gcloud` setup on your machine. 

```bash
PROJECT_ID=<REPLACE_WITH_YOUR_PROJECT_ID>
REGION=<REPLACE_WITH_YOUR_GCP_REGION_NAME>
AR_REPO=<REPLACE_WITH_YOUR_AR_REPO_NAME>
SERVICE_NAME=chat-gradio-app
gcloud artifacts repositories create $AR_REPO --location=$REGION --repository-format=Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME
gcloud run deploy $SERVICE_NAME --port 8080 --image $REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE_NAME --allow-unauthenticated --region=$REGION --platform=managed  --project=$PROJECT_ID
```

Alternately, if you are in VS Code using the Cloud Code Extension, you can deploy the application directly to Cloud Run by:

Click on Cloud Code extension in the bar at the bottom.
Select Deploy to Cloud Run.
Go ahead with the defaults. Modify them accordingly if you'd like.
Click on Deploy.
This will take a few minutes and the result will be the Cloud Run Service URL that you can use directly in the web browser to access the application.
