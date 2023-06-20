import streamlit as st
import pandas as pd 

@st.cache_data()
def get_predefined_vector_store():
    return pd.read_pickle('./temp/vector_store_data_typedf_20230426_alphabet_10Q.pkl')


def reset_session() -> None:
    """_summary_: Resets the session state to default values.
    """
    clear_chat()
    st.session_state['vector_store_data_chromadb_object'] = ""
    st.session_state['vector_store_data_typedf'] = []
    st.session_state['processed_data_list'] = []
    st.session_state['file_upload_flag'] = False
    st.session_state['processor_version'] = "OpenSource(PyPDF2)"
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
    st.session_state['vector_db_context'] = ""
    st.session_state['vector_db_matched_db'] = pd.DataFrame()
    st.session_state['vector_db_source'] = ""
    st.session_state['answer'] = ""
    st.session_state['rewritten_answer']  = ""
    st.session_state['rewritten_bullet_answer']  = ""
    st.session_state['question'] = ""
    st.session_state['focused_answer'] = ""
    st.session_state['chat_model'] = ""
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['context'] = ""
    st.session_state['example'] = []
    st.session_state['temperature'] = []
    st.session_state['focused_answer_explainer'] = ""
    st.session_state['focused_citation_df'] = pd.DataFrame()
    st.session_state['document_summary_mapreduce'] = ""
    st.session_state['demo_mode'] = "DemoMode"
    st.session_state['demo_mode_vector_store_data_typedf'] = get_predefined_vector_store()
    st.session_state['vector_store_flag_demo'] = False
    
def hard_reset_session() -> None: 
    st.session_state = {states : [] for states in st.session_state}


def create_session_state():
    """
    Creating session states for the app.
    """
    if "demo_mode_vector_store_data_typedf" not in st.session_state:
        st.session_state['demo_mode_vector_store_data_typedf'] = get_predefined_vector_store()
    if 'vector_store_flag_demo' not in st.session_state:
        if not st.session_state['demo_mode_vector_store_data_typedf'].empty:
            st.session_state['vector_store_flag_demo'] = True
        else:
            st.session_state['vector_store_flag_demo'] = False
    if "demo_mode" not in st.session_state:
        st.session_state['demo_mode'] = "DemoMode"
    if "document_summary_mapreduce" not in st.session_state:
        st.session_state['document_summary_mapreduce'] = ""
    if "focused_citation_df" not in st.session_state:
        st.session_state['focused_citation_df'] = pd.DataFrame()
    if "focused_answer_explainer" not in st.session_state:
        st.session_state['focused_answer_explainer'] = ""
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = []
    if 'debug_mode' not in st.session_state:
        st.session_state['debug_mode'] = False
    if 'chat_input' not in st.session_state:
        st.session_state['chat_input'] = ''
    if 'context' not in st.session_state:
        st.session_state['context'] = ''
    if 'example' not in st.session_state:
        st.session_state['example'] = []
    if 'chat_model' not in st.session_state:
        st.session_state['chat_model'] = ""
    if 'focused_answer' not in st.session_state:
        st.session_state['focused_answer'] = ""
    if 'question' not in st.session_state:
        st.session_state['question']  = ""
    if 'rewritten_bullet_answer' not in  st.session_state:
        st.session_state['rewritten_bullet_answer']  = ""
    if 'answer' not in st.session_state:
        st.session_state['vector_db_context'] = ""
    if 'rewritten_answer' not in st.session_state:
        st.session_state['rewritten_answer'] = ""
    if 'vector_db_context' not in st.session_state:
        st.session_state['vector_db_context'] = ""
    if 'vector_db_matched_db' not in st.session_state:
        st.session_state['vector_db_matched_db'] = pd.DataFrame()
    if 'vector_db_source' not in st.session_state:
        st.session_state['vector_db_source'] = ""
    if 'vector_store_data_chromadb_object' not in st.session_state:
        st.session_state['vector_store_data_chromadb_object'] = ""
    if 'vector_store_data_typedf' not in st.session_state:
        st.session_state['vector_store_data_typedf'] = pd.DataFrame()
    if 'processed_data_list' not in st.session_state:
        st.session_state['processed_data_list'] = []
    if 'file_upload_flag' not in st.session_state:
        st.session_state['file_upload_flag'] = False
    if 'processor_version' not in st.session_state:
        st.session_state['processor_version'] = "OpenSource(PyPDF2)"
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


def clear_chat() -> None:
    st.session_state['chat_model'] = ""
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['context'] = ""
    st.session_state['example'] = []
    st.session_state['temperature'] = []

