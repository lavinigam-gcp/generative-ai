import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from src.vertex import *


def get_collection_for_chromadb_fromData(data:list = []):
    document = []
    metadata = []
    id = []
    for index, eachpacket in enumerate(data):
        # if index < 5: 
        document.append(eachpacket['content'])
        metadata.append({k:v for k,v in eachpacket.items() if k not in "content"})
        id.append("id_"+str(index+1))
    # st.write(document, metadata, id)
    return document, metadata, id


class VertexPaLMEmbeddings(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        # embed the documents somehow
        # print(texts)
        final_emb = []
        for eachtext in texts:
        #   embeddings = embedding_model.get_embeddings()
        #   embs =  [each.values for each in embeddings][0]
          final_emb.append(embedding_model_with_backoff([eachtext]))
        return final_emb


def get_chromadb_create_collection_vectorstore(data):
    client = chromadb.Client()
    PaLMEmbeddingFunction  = VertexPaLMEmbeddings()
    document, metadata, id = get_collection_for_chromadb_fromData(data)
    # st.write(document[0],metadata[0],id[0])
    palm_collection = client.create_collection(name=st.session_state['env_config']['chromadb']['collection_name'],
                                                embedding_function=PaLMEmbeddingFunction)
    with st.spinner('Building vector store with chromaDB....'):
        palm_collection.add(
            documents=document,
            metadatas=metadata,
            ids=id
            
        )
        st.write("Vectore Store Built.......")

        if palm_collection.count():
            return palm_collection