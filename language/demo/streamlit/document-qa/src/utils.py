import streamlit as st
from streamlit_chat import message
import pandas as pd
import ast
from PyPDF2 import PdfReader
import os
import re
import numpy as np
from src.vertex import *
from src.vector_store_chromadb import *
# @st.cache_data()
# def get_env_config():
#     path = Path("./src/env.yaml")
#     yaml = YAML(typ='safe')
#     return yaml.load(path)


def reset_session() -> None:
    """_summary_: Resets the session state to default values.
    """
    st.session_state['vector_store_data_chromadb_object'] = ""
    st.session_state['vector_store_data_typedf'] = []
    st.session_state['processed_data_list'] = []
    st.session_state['file_upload_flag'] = False
    st.session_state['processor_version'] = "OpenSource"
    st.session_state['vector_db'] = "pandas"
    st.session_state['process_doc'] = False
    st.session_state['interface'] = "TextGeneration API"
    st.session_state['temperature'] = 0.0
    st.session_state['token_limit'] = 256
    st.session_state['top_k'] = 40
    st.session_state['top_p'] = 0.8
    st.session_state['debug_mode'] = False
    st.session_state['prompt'] = []
    st.session_state['response'] = []
    st.session_state['vector_store_flag'] = False
    st.session_state['process_doc'] = False
    st.session_state['chunk_size'] = 500
    st.session_state['top_sort_value'] = 5
    st.session_state['vector_db_pandas_context'] = ""
    st.session_state['vector_db_pandas_matched_db'] = pd.DataFrame()
    st.session_state['vector_db_pandas_source'] = ""
    st.session_state['answer'] = ""
    st.session_state['rewritten_answer']  = ""
    st.session_state['rewritten_bullet_answer']  = ""
    
def hard_reset_session() -> None: 
    st.session_state = {states : [] for states in st.session_state}


def create_session_state():
    """
    Creating session states for the app.
    """
    if 'rewritten_bullet_answer' not in  st.session_state:
        st.session_state['rewritten_bullet_answer']  = ""
    if 'answer' not in st.session_state:
        st.session_state['vector_db_pandas_context'] = ""
    if 'rewritten_answer' not in st.session_state:
        st.session_state['rewritten_answer'] = ""
    if 'vector_db_pandas_context' not in st.session_state:
        st.session_state['vector_db_pandas_context'] = ""
    if 'vector_db_pandas_matched_db' not in st.session_state:
        st.session_state['vector_db_pandas_matched_db'] = pd.DataFrame()
    if 'vector_db_pandas_source' not in st.session_state:
        st.session_state['vector_db_pandas_source'] = ""

    if 'vector_store_data_chromadb_object' not in st.session_state:
        st.session_state['vector_store_data_chromadb_object'] = ""
    if 'vector_store_data_typedf' not in st.session_state:
        st.session_state['vector_store_data_typedf'] = pd.DataFrame()
    if 'processed_data_list' not in st.session_state:
        st.session_state['processed_data_list'] = []
    if 'file_upload_flag' not in st.session_state:
        st.session_state['file_upload_flag'] = False
    if 'processor_version' not in st.session_state:
        st.session_state['processor_version'] = "OpenSource"
    if 'vector_db' not in st.session_state:
        st.session_state['vector_db'] = "pandas"
    if 'interface' not in st.session_state:
        st.session_state['interface'] = "TextGeneration API"
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.0
    if 'token_limit' not in st.session_state:
        st.session_state['token_limit'] = 256
    if 'top_k' not in st.session_state:
        st.session_state['top_k'] = 40
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 0.8
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = []
    if 'response' not in st.session_state:
        st.session_state['response'] = []
    if 'vector_store_flag' not in st.session_state:
        st.session_state['vector_store_flag'] = False
    if 'process_doc' not in st.session_state:
        st.session_state['process_doc'] = False
    if 'chunk_size' not in st.session_state:
        st.session_state['chunk_size'] = 500
    if 'top_sort_value' not in st.session_state:
        st.session_state['top_sort_value'] = 5


    
def create_data_packet(file_name, file_type, page_number, chunk_number, file_content):
    """Creating a simple dictionary to store all information (content and metadata)
    extracted from the document"""
    data_packet = {}
    data_packet["file_name"] = file_name
    data_packet["file_type"] = file_type
    data_packet["page_number"] = page_number
    data_packet["chunk_number"] =  chunk_number
    data_packet["content"] = file_content
    return data_packet

def get_chunks_iter(text, maxlength):
    """
    Get chunks of text, each of which is at most maxlength characters long.

    Args:
        text: The text to be chunked.
        maxlength: The maximum length of each chunk.

    Returns:
        An iterator over the chunks of text.
    """
    start = 0
    end = 0
    final_chunk = []
    while start + maxlength < len(text) and end != -1:
        end = text.rfind(" ", start, start + maxlength + 1)
        final_chunk.append(text[start:end])
        start = end + 1
    final_chunk.append(text[start:])
    return final_chunk


# function to apply "get_chunks_iter" function on each row of dataframe.
# currently each row here for file_type=pdf is content of each page and for other file_type its the whole document.
def split_text(row):
    """_summary_: Splits the text into chunks of given size.

    Args:
        row (_type_): each row of the pandas dataframe through apply function.

    Returns:
        _type_: list of chunks of text.
    """
    chunk_iter = get_chunks_iter(row, chunk_size)
    return chunk_iter

def get_dot_product(row):
    return np.dot(row, query_vector)


def get_data_loading(documents, method:str = "OpenSource",
                     chunk_size_value:int=2000):
    if method == "OpenSource":
        final_data = []
        with st.spinner('Loading documents using PyPDF2 and `open` method....'):
            for eachdoc in documents:
                file_name, file_type = os.path.splitext(eachdoc.name)
                if file_type == ".pdf":
                    # loading pdf files, with page numbers as metadata.
                    reader = PdfReader(eachdoc)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            text_chunks = get_chunks_iter(text, chunk_size_value)
                            for chunk_number, chunk_content in enumerate(text_chunks):
                                packet = create_data_packet(
                                    file_name, file_type,page_number=int(i + 1), chunk_number = chunk_number+1,
                                    file_content=chunk_content
                                )

                                final_data.append(packet)
        return final_data
    elif method == "DocumentAI":
        final_data = []
        with st.spinner('Loading documents using DocumentAI....'):
            for eachdoc in documents:
                bytes_data = eachdoc.getvalue()
                # st.write(bytes_data)
                file_name, file_type = os.path.splitext(eachdoc.name)
                print("processing...",file_name)
                if file_type == ".pdf":
                    document=process_document(st.session_state['env_config']['gcp']['PROJECT_ID'], 
                                                st.session_state['env_config']['documentai']['DOCAI_LOCATION'],
                                                st.session_state['env_config']['documentai']['DOCAI_PROCESSOR_ID'],
                                                st.session_state['env_config']['documentai']['DOCAI_PROCESSOR_VERSION'],
                                                bytes_data,
                                                mime_type=st.session_state['env_config']['documentai']['mime_type'])
                    if document:
                        document_chunks = get_chunks_iter(document.text, chunk_size_value)
                        for chunk_number, chunk_content in enumerate(document_chunks):
                            packet = create_data_packet(
                                file_name, file_type,page_number="NotAvailable", chunk_number = chunk_number+1,
                                file_content=chunk_content
                            )

                            final_data.append(packet)
        return final_data
    
def get_data_vectors(data:list, vector_db_choice:str = "Pandas"
                     ):
    if vector_db_choice == "Pandas":
        #Converting data to Pandas
        pdf_data = pd.DataFrame.from_dict(data)
        # st.write(pdf_data)
        pdf_data = pdf_data.sort_values(
            by=["file_name"]
        )  # sorting the datafram by filename and page_number
        pdf_data.reset_index(inplace=True, drop=True)

        pdf_data["types"] = pdf_data["content"].apply(
            lambda x: type(x)
            )
        # st.write(pdf_data.head())
        # st.write(pdf_data.dtypes)
        #For testing & Debug. Remove or comment. 
        # pdf_data = pdf_data.head(10)
        with st.spinner('Building vectors of the chunk..beep boop..taking time....'):
            pdf_data["embedding"] = pdf_data["content"].apply(
            lambda x: embedding_model_with_backoff([x])
            )
            pdf_data["embedding"] = pdf_data.embedding.apply(np.array)
            
        st.write("Vectore Store of your documents is created in Pandas.....")
        # st.write(pdf_data.head())
        st.session_state['vector_store_flag'] = True
        st.session_state['vector_store_data_typedf'] = pdf_data
        return st.session_state['vector_store_data_typedf']
    elif vector_db_choice == "Chroma":
        st.write("Chroma VectorDB.......")
        # st.write(data[0])
        # subset_data = data[:5] #only sending first 5 data
        chroma_collection_object = get_chromadb_create_collection_vectorstore(data)
        if chroma_collection_object:
            st.session_state['vector_store_flag'] = True
            st.session_state['vector_store_data_chromadb_object'] = chroma_collection_object
            return st.session_state['vector_store_data_chromadb_object']


def get_filter_context_from_vectordb(vector_db_choice:str = "Pandas",
                                     question: str = "",
                                     sort_index_value:int = 3):
    if vector_db_choice == "Pandas" and st.session_state['vector_store_flag'] and not st.session_state['vector_store_data_typedf'].empty:
        global query_vector
        query_vector = np.array(embedding_model_with_backoff([question]))
        top_matched_score = (
            st.session_state['vector_store_data_typedf']["embedding"]
            .apply(get_dot_product)
            .sort_values(ascending=False)[:sort_index_value]
        )
        top_matched_df = st.session_state['vector_store_data_typedf'][st.session_state['vector_store_data_typedf'].index.isin(top_matched_score.index)]
        top_matched_df = top_matched_df[['file_name','page_number','chunk_number','content']]
        top_matched_df['confidence_score'] = top_matched_score
        top_matched_df.sort_values(by=['confidence_score'], ascending=False,inplace=True)
        context = "\n".join(
            st.session_state['vector_store_data_typedf'][st.session_state['vector_store_data_typedf'].index.isin(top_matched_score.index)]["content"].values
        )
        source = f"""filenames: {",".join(top_matched_df['file_name'].value_counts().index) },
                pages: {top_matched_df['page_number'].unique()} , 
                chunk: {top_matched_df['chunk_number'].unique()}
                """
        return (context, top_matched_df,source)
    elif vector_db_choice == "Chroma" and st.session_state['vector_store_data_chromadb_object']:
        results = st.session_state['vector_store_data_chromadb_object'].query(query_texts=question,
                                        n_results=sort_index_value
                                        )
        metadata_df = pd.DataFrame(results["metadatas"][0])
        id_df = pd.DataFrame(results["ids"][0],columns=["id"])
        document_df = pd.DataFrame(results["documents"][0],columns=["content"])
        distance_df = pd.DataFrame(results["distances"][0],columns=["confidence_score"])
        top_matched_df = pd.concat([metadata_df, id_df, document_df, distance_df],axis=1)
        top_matched_df.sort_values(by=['confidence_score'], ascending=False,inplace=True)

        context = " ".join(top_matched_df["content"])

        source = f"""filenames: {",".join(top_matched_df['file_name'].value_counts().index) },
                pages: {top_matched_df['page_number'].unique()} , 
                chunk: {top_matched_df['chunk_number'].unique()}
                """
        
        return (context, top_matched_df,source)


    


# @st.cache_data
# def read_documents(documents,chunk_size_value=2000, sample=True, sample_size=10):
#     """_summary_: Reads the documents and creates a pandas dataframe with all the content and metadata.
#     cleaning the text and splitting the text into chunks of given size. creating a vector store of the chunks.

#     Args:
#         documents (_type_): list of documents uploaded by the user.
#         chunk_size_value (_type_, optional): size of each chunk. Defaults to 2000.
#         sample (bool, optional): whether to create a sample vector store or not. Defaults to True.
#         sample_size (int, optional): size of the sample vector store. Defaults to 10.

#     Returns:
#         _type_: pandas dataframe with all the content and metadata.
    
#     """
#     final_data = []
#     with st.spinner('Loading documents and putting them in pandas dataframe.....'):
#         for eachdoc in documents:
#             file_name, file_type = os.path.splitext(eachdoc.name)
#             if file_type == ".pdf":
#                 # loading pdf files, with page numbers as metadata.
#                 reader = PdfReader(eachdoc)
#                 for i, page in enumerate(reader.pages):
#                     text = page.extract_text()
#                     if text:
#                         packet = create_data_packet(
#                             file_name, file_type, page_number=int(i + 1), file_content=text
#                         )

#                         final_data.append(packet)
#             elif file_type == ".txt":
#                 # loading other file types
#                 # st.write(eachdoc)
#                 text = eachdoc.read().decode("utf-8")
#                 # text = textract.process(bytes_data).decode("utf-8")
#                 packet = create_data_packet(
#                     file_name, file_type, page_number=-1, file_content=text
#                 )
#                 final_data.append(packet)
        
#         # st.write(final_data)
#         pdf_data = pd.DataFrame.from_dict(final_data)
#         # st.write(pdf_data)
#         pdf_data = pdf_data.sort_values(
#             by=["file_name", "page_number"]
#         )  # sorting the datafram by filename and page_number
#         pdf_data.reset_index(inplace=True, drop=True)

#     with st.spinner('Splitting data into chunks and cleaning the text...'):
#         global chunk_size
#         # you can define how many words should be there in a given chunk.
#         chunk_size = chunk_size_value

#         pdf_data["content"] = pdf_data["content"].apply(
#         lambda x: re.sub("[^A-Za-z0-9]+", " ", x)
#                         )

#         # Apply the chunk splitting logic here on each row of content in dataframe.
#         pdf_data["chunks"] = pdf_data["content"].apply(split_text)
#         # Now, each row in 'chunks' contains list of all chunks and hence we need to explode them into individual rows.
#         pdf_data = pdf_data.explode("chunks")

#         # Sort and reset index
#         pdf_data = pdf_data.sort_values(by=["file_name", "page_number"])
#         pdf_data.reset_index(inplace=True, drop=True)

#     with st.spinner('Building vectors of the chunk..beep boop..taking time....'):
#         if sample:
#             pdf_data_sample = pdf_data.sample(sample_size)
#         else:
#             pdf_data_sample = pdf_data.copy()
        
#         pdf_data_sample["embedding"] = pdf_data_sample["content"].apply(
#         lambda x: embedding_model_with_backoff([x])
#         )
#         pdf_data_sample["embedding"] = pdf_data_sample.embedding.apply(np.array)
        
#     st.write("Vectore Store of your documents is created.....")
#     return pdf_data_sample



# def get_context_from_question(question, vector_store, sort_index_value=2):
#     global query_vector
#     query_vector = np.array(embedding_model_with_backoff([question]))
#     top_matched = (
#         vector_store["embedding"]
#         .apply(get_dot_product)
#         .sort_values(ascending=False)[:sort_index_value]
#         .index
#     )
#     top_matched_df = vector_store[vector_store.index.isin(top_matched)][
#         ["file_name", "page_number", "content","chunks"]
#     ]
#     context = "\n".join(
#         vector_store[vector_store.index.isin(top_matched)]["chunks"].values
#     )
#     source = f"""filenames: {",".join(top_matched_df['file_name'].value_counts().index) },
#               pages: {top_matched_df['page_number'].unique()}
#               """
#     return context, top_matched_df,source
# elif file_type == ".txt":
#     # loading other file types
#     # st.write(eachdoc)
#     text = eachdoc.read().decode("utf-8")
#     # text = textract.process(bytes_data).decode("utf-8")
#     if text:
#         text_chunks = get_chunks_iter(text, chunk_size)
#         for chunk_number, chunk_content in enumerate(text_chunks):
#             packet = create_data_packet(
#                 file_name, file_type,page_number=None, chunk_number = chunk_number+1,
#                 file_content=chunk_content
#             )

#             final_data.append(packet)
#     final_data.append(packet)