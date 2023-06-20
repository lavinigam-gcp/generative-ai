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

def chat_input_submit():
    st.session_state.chat_input = st.session_state.chat_widget
    st.session_state.chat_widget = ''

def clear_duplicate_data():
    for i in range(len(st.session_state['past'])-1,0,-1):
        if st.session_state['past'][i][i] == st.session_state['past'][i-1][i-1]:
            del st.session_state['past'][i]
            del st.session_state['generated'][i]
    
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


def get_data_loading(documents, method:str = "OpenSource(PyPDF2)",
                     chunk_size_value:int=2000):
    if method == "OpenSource(PyPDF2)":
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
        # st.write(data[0])
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
        #Storing vector store in pickle file for demo mode
        # st.session_state['vector_store_data_typedf'].to_pickle("./temp/vector_store_data_typedf_20230426_alphabet_10Q.pkl")
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

def get_dot_product(row):
    # st.write(type(row))
    # st.write(type(query_vector))
    return np.dot(row, query_vector)


def get_filter_context_from_vectordb(vector_db_choice:str = "Pandas",
                                     question: str = "",
                                     sort_index_value:int = 3):
    if vector_db_choice == "Pandas" and st.session_state['vector_store_flag'] and not st.session_state['vector_store_data_typedf'].empty:
        # st.write(st.session_state['vector_store_data_typedf'])
        # st.write(st.session_state['vector_store_data_typedf'].dtypes)
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
    elif vector_db_choice == "DemoMode" and st.session_state['vector_store_flag_demo'] and not st.session_state['demo_mode_vector_store_data_typedf'].empty:
        # st.write(st.session_state['vector_store_data_typedf'])
        # st.write(st.session_state['vector_store_data_typedf'].dtypes)
        # global query_vector
        st.session_state['demo_mode_vector_store_data_typedf']["embedding"] = st.session_state['demo_mode_vector_store_data_typedf'].embedding.apply(np.array)
        query_vector = np.array(embedding_model_with_backoff([question]))
        top_matched_score = (
            st.session_state['demo_mode_vector_store_data_typedf']["embedding"]
            .apply(get_dot_product)
            .sort_values(ascending=False)[:sort_index_value]
        )
        top_matched_df = st.session_state['demo_mode_vector_store_data_typedf'][st.session_state['demo_mode_vector_store_data_typedf'].index.isin(top_matched_score.index)]
        top_matched_df = top_matched_df[['file_name','page_number','chunk_number','content']]
        top_matched_df['confidence_score'] = top_matched_score
        top_matched_df.sort_values(by=['confidence_score'], ascending=False,inplace=True)
        context = "\n".join(
            st.session_state['demo_mode_vector_store_data_typedf'][st.session_state['demo_mode_vector_store_data_typedf'].index.isin(top_matched_score.index)]["content"].values
        )
        source = f"""filenames: {",".join(top_matched_df['file_name'].value_counts().index) },
                pages: {top_matched_df['page_number'].unique()} , 
                chunk: {top_matched_df['chunk_number'].unique()}
                """
        return (context, top_matched_df,source)

# function to pass in the apply function on dataframe to extract answer for specific question on each row.
def get_answer(df):
    prompt = f"""Answer the question as precise as possible using the provided context. If the answer is
                 not contained in the context, say "answer not available in context" \n\n
                  Context: \n {df['content']}?\n
                  Question: \n {st.session_state['question']} \n
                  Answer:
            """

    pred = text_generation_model_with_backoff(prompt=prompt)
    return pred


def get_focused_map_reduce_without_embedding(vector_db_choice:str = "Pandas",
                                     question: str = "",
                                     sort_index_value:int = 20):
    # st.write("size of vector store: ",st.session_state['vector_store_data_typedf'].shape[0])
    # st.write("sort_index_value:",sort_index_value)
    # st.write("vector store len: ",st.session_state['vector_store_data_typedf'].shape[0])

    if vector_db_choice == "Pandas":
        if sort_index_value < st.session_state['vector_store_data_typedf'].shape[0]:
            sort_index_value = sort_index_value
        else:
            sort_index_value = int((st.session_state['vector_store_data_typedf'].shape[0])/2)

    context, top_matched_df, source = get_filter_context_from_vectordb(vector_db_choice ,
                                    {st.session_state['question']} ,
                                    sort_index_value
                                    )
    top_matched_df["predicted_answer"] = top_matched_df.apply(get_answer, axis=1)
    # st.write(top_matched_df)
    context_map_reduce = [eachanswer
                            for eachanswer in top_matched_df["predicted_answer"].values
                            if eachanswer.lower() != "answer not available in context"
                        ]
    if context_map_reduce:
        joined = ".".join(context_map_reduce)
        context_new = get_text_generation(prompt=f"rewrite the text properly as a professional text with proper english: {joined}")
        # st.write("context_map_reduce::::",context_map_reduce)
        prompt = f"""Answer the question using the provided context. If the answer is
                        not contained in the context, say "precise answers not available in context" \n\n
                        Context: \n {context_new}?\n
                        Question: \n {st.session_state['question']} \n
                        Answer:
                    """
        # print("the words in the prompt: ", len(prompt))
        # print("PaLM Predicted:", generation_model.predict(prompt).text)
        focused_answer = get_text_generation(prompt=prompt)
    else:
        context_new = "Since nothing matched, no raw context available"
        focused_answer = "precise answers not available in context"
    return focused_answer,context_new, top_matched_df
    

# Define a function to create a summary of the summaries
def reduce_summary(summary, prompt_template):

    # Concatenate the summaries from the inital step
    concat_summary = "\n".join(summary)

    # Create a prompt for the model using the concatenated text and a prompt template
    prompt = prompt_template.format(text=concat_summary)

    # Generate a summary using the model and the prompt
    summary = text_generation_model_with_backoff(prompt=prompt, max_output_tokens=1024)

    return summary

def get_map_reduce_summary():
    if st.session_state['processed_data_list']:
        if len(st.session_state['processed_data_list']) > 10:
            st.session_state['processed_data_list'] = st.session_state['processed_data_list'][:10]

        initial_prompt_template = """
                                Write a concise summary of the following text delimited by triple backquotes.

                                ```{text}```

                                CONCISE SUMMARY:
                                """

        final_prompt_template = """
                                Write a concise summary of the following text delimited by triple backquotes.
                                Return your response in bullet points which covers the key points of the text.

                                ```{text}```

                                BULLET POINT SUMMARY:
                            """
        # Create an empty list to store the summaries
        initial_summary = []
        for eachblock in st.session_state['processed_data_list']:

             # Create a prompt for the model using the extracted text and a prompt template
            prompt = initial_prompt_template.format(text=eachblock['content'])

            # Generate a summary using the model and the prompt
            summary = text_generation_model_with_backoff(prompt=prompt, max_output_tokens=1024)

            # Append the summary to the list of summaries
            initial_summary.append(summary)
        # Use defined `reduce` function to summarize the summaries
        summary = reduce_summary(initial_summary, final_prompt_template)
        return summary
    else: 
        st.write("The document has not been loaded and processed. Kindly do that....")
        return False
