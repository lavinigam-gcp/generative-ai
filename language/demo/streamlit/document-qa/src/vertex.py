from vertexai.preview.language_models import TextGenerationModel,TextEmbeddingModel
import vertexai
import streamlit as st
import backoff
from google.api_core.exceptions import ResourceExhausted
from google.cloud import documentai
from google.api_core.exceptions import NotFound
from google.cloud.documentai import DocumentProcessorServiceClient
from google.cloud import documentai  # type: ignore
from google.cloud.documentai import DocumentProcessorServiceClient
from google.api_core.client_options import ClientOptions
from google.cloud.documentai import Processor
from google.cloud.documentai import ProcessorType
from google.cloud.documentai import ProcessRequest
from google.cloud.documentai import RawDocument
from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore


vertexai.init(project=st.session_state['env_config']['gcp']['PROJECT_ID'], 
              location=st.session_state['env_config']['gcp']['LOCATION_NAME'])


@st.cache_resource
def get_model():
    generation_model = TextGenerationModel.from_pretrained("text-bison@001")
    return generation_model

@st.cache_resource
def get_embedding_model():
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    return embedding_model

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def text_generation_model_with_backoff(**kwargs):
    return get_model().predict(**kwargs).text

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def embedding_model_with_backoff(text=[]):
    embeddings = get_embedding_model().get_embeddings(text)
    return [each.values for each in embeddings][0]


def get_text_generation(prompt="",  **parameters):
    generation_model = get_model()
    response = generation_model.predict(prompt=prompt, **parameters
                                        )

    return response.text

def process_document(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    file_content: bytes,
    mime_type: str,
) -> documentai.Document:
    # You must set the api_endpoint if you use a location other than 'us'.
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the processor version
    # e.g. projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}
    # You must create processors before running sample code.
    name = client.processor_version_path(
        project_id, location, processor_id, processor_version
    )

    # Read the file into memory
    # with open(file_path, "rb") as image:
    #     image_content = image.read()

    # Load Binary Data into Document AI RawDocument Object
    raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)

    # Configure the process request
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    result = client.process_document(request=request)

    return result.document

