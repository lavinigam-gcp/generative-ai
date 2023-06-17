import streamlit as st
st.set_page_config(
    page_title="Question Answering using Vertex PaLM API",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This app shows you how to use Vertex PaLM Text Generator API on your custom documents"
    }
)
import src.html_component as html_comp
from ruamel.yaml import YAML
from pathlib import Path
from rouge_score import rouge_scorer

@st.cache_data()
def get_env_config():
    st.write("Loading env.yaml")
    path = Path("./src/env.yaml")
    yaml = YAML(typ='safe')
    return yaml.load(path)
    
if 'env_config' not in st.session_state:    
    st.session_state['env_config'] = get_env_config()
    
from src.utils import *
from streamlit_chat import message
from PIL import Image
from io import StringIO
import base64

# ##creating session states
create_session_state()

@st.cache_data()
def get_image_displayed(path):
    image = Image.open(path)
    st.image(image)

def change_file_upload_flag():
    st.session_state['file_upload_flag'] = False
# './image/palm.jpg'
#Page Header Image and Text
# image = Image.open('./image/palm.jpg')
# st.image(image)
get_image_displayed(path='./image/palm.jpg')
st.markdown("<h1 style='text-align: center; color: darkgreen;'>DOCY - Enterprise Search with PaLM</h1>", unsafe_allow_html=True)

# Left side of APP. 
#Sidebar 
with st.sidebar:
    # sidebar_image = Image.open('./image/bot2.png')
    # st.image(sidebar_image)
    get_image_displayed(path='./image/bot2.png')
    st.markdown("<h1 style='text-align: center; color: blue;'>Customize your Docy</h1>", unsafe_allow_html=True)


    # Debug Mode
    debug_mode_choice = st.radio("Debug mode (keep it False)", (False,True),horizontal=True)
    st.session_state['debug_mode'] = debug_mode_choice
    # Select the processor version [OpenSource, DocumentAI]
    processor_version = st.radio("Select the document loader", ( "OpenSource", "DocumentAI"),horizontal=True)
    st.session_state['processor_version'] = processor_version
    if st.session_state['processor_version'] == "DocumentAI":
        st.write("*As of now you can only upload PDF documents with limit of upto 14 pages. Multiple document supported.")
    elif st.session_state['processor_version'] == "OpenSource":
        st.write("*As of now you can only upload PDF documents. Multiple document supported. Using PyPDF2")
        
    # Select Vector DB [Pandas, Faiss, Chroma, MatchingEngine]
    vector_db = st.radio("Select the Vector DB", ( "Pandas", "Chroma", "FAISS","MatchingEngine"))
    st.session_state['vector_db'] = vector_db
    if st.session_state['vector_db'] == "FAISS" or st.session_state['vector_db'] == "MatchingEngine":
        st.write("Coming Soon....")

    # # Interface [TextGeneration API, Chat API]
    # interface = st.radio("Select the Interface", ( "TextGeneration API", "Chat API"),horizontal=True)
    # st.session_state['interface'] = interface

    # Selecting the chunk size
    chunk_size_value = st.number_input("Chunk Size:",value=500,min_value=200, max_value=10000,step=200)
    st.session_state['chunk_size'] = chunk_size_value

    # Select Top N results
    top_match_result_value = st.number_input("How many result to pick?",value=3)
    st.session_state['top_sort_value'] = top_match_result_value

    if not st.session_state['file_upload_flag']:
        user_docs = st.file_uploader(
                "Upload your documents. PDF and TXT Only", accept_multiple_files=True, type=['pdf','txt'],
                on_change = change_file_upload_flag)
        

    # Loading & Processing Documents as per the selected Method
    if not st.session_state['process_doc']:
        if st.button("Process"):
            
            # Loading, chunking the documents 
            with st.spinner("Loading and Chunking input files...this will take time..."):
                final_data = get_data_loading(user_docs, method = st.session_state['processor_version'],
                                            chunk_size_value = st.session_state['chunk_size'])
                # vector_store = cache_vector_store(final_data)
                st.session_state['processed_data_list'] = final_data
                
                if final_data:
                    st.session_state['process_doc'] = True
                else: 
                    st.session_state['process_doc'] = False

            #Building the vector store
            with st.spinner("Building your indexes with your VectorDB choice...this will take time..."):
                vector_store_return_obj = get_data_vectors(final_data, 
                                                vector_db_choice = st.session_state['vector_db']
                                                )
                
                st.write("Vector DB Status: ", st.session_state['vector_store_flag'])
                #     st.write(st.session_state['vector_store_data_typedf'])
                #     st.write(st.session_state['vector_store_data_chromadb_object'])


            #Getting Nearest Neighbors to user Query
            # Happening inside container 
            
    else:
        st.write("Your document is already loaded, processed and pushed to Vector DB. Ask away your question -->")

    if st.button("Reset Session"):
        reset_session()

    get_image_displayed(path='./image/bot3.png')

# Right side of Screen [ Query Block ]
with st.container():
    if st.session_state['debug_mode']:
        try:
            st.write("You have enabled Debug Mode. This is for Admin. If you did this by mistake, select False.")
            st.write(st.session_state)
            
            # st.write(st.session_state['processed_data_list'][0])
        except IndexError:
            st.write("IndexError: No data to display, upload the documents and process it. Or Reset Session.")

    if st.session_state['vector_store_flag']:
        
        question = st.text_input('What would you like to ask the documents?')
        # get the custom relevant chunks from all the chunks in vector store.
        if question:
            if st.session_state['vector_db'] == "Pandas":
                context, top_matched_df, source = get_filter_context_from_vectordb(vector_db_choice = st.session_state['vector_db'],
                                     question = question,
                                     sort_index_value =  st.session_state['top_sort_value'])
            elif st.session_state['vector_db'] == "Chroma":
                context, top_matched_df, source = get_filter_context_from_vectordb(vector_db_choice = st.session_state['vector_db'],
                                     question = question,
                                     sort_index_value =  st.session_state['top_sort_value'])
                
            if not top_matched_df.empty: 
                st.session_state['vector_db_pandas_context'] =  context
                st.session_state['vector_db_pandas_matched_db'] = top_matched_df
                st.session_state['vector_db_pandas_source'] = source
                st.markdown("<h3 style='text-align: center; color: black;'>Here's the answer from document</h3>", unsafe_allow_html=True)
                # st.write("Here's the answer from document: ")
                tab1, tab2, tab3, tab4 = st.tabs(["Grounded", "PaLMReWrite","PaLMBulletSummary","Evaluation"])
                
                with tab1:
                    prompt = f""" Answer the question as precise as possible using the provided context. \n\n
                        Context: \n {st.session_state['vector_db_pandas_context']}?\n
                        Question: \n {question} \n
                        Answer:
                    """
                    answer = get_text_generation(prompt=prompt,temperature=0.0,max_output_tokens=1024)
                    st.session_state['answer'] = answer
                    st.write(st.session_state['answer'])
                    
                with tab2:
                    prompt = f"""can you write a essay based on the following text. 
                    The essay should explain the points given in the {{text}}. 
                    Do not add anything on your own and only pick context from the {{text}}: \n text: \t {st.session_state['answer']}
                    """
                    rewritten_answer = get_text_generation(prompt=prompt,temperature=0.5,max_output_tokens=1024)
                    st.session_state['rewritten_answer']  = rewritten_answer
                    st.write(st.session_state['rewritten_answer'])
                with tab3:
                    prompt = f"""Write a concise summary of the following text delimited by triple backquotes.
                                Return your response in bullet points which covers the key points of the text.

                            ```{st.session_state['answer']}```

                            BULLET POINT SUMMARY: \n 
                            """
                    rewritten_bullet_answer = get_text_generation(prompt=prompt,temperature=0.5,max_output_tokens=1024)
                    st.session_state['rewritten_bullet_answer']  = rewritten_bullet_answer
                    st.write(st.session_state['rewritten_bullet_answer'])
                with tab4:
                    st.write(""":red[ROUGE] (Recall-Oriented Understudy for Gisting Evaluation) is a set of 
                    metrics for evaluating automatic summarization and machine translation software in natural language processing.""")
                    st.write(""" \n\n
                    * :darkblue[The ROUGE-1:] It measures the overlap of unigrams between the system summary and the reference summary. \n
                    * :darkblue[The ROUGE-L:] It measures the longest common subsequence (LCS) between the system summary and the reference summary. \n
                    * :darkblue[Precision:] The fraction of the words in the system summary that are also in the reference summary. \n
                    * :darkblue[Recall:] The fraction of the words in the reference summary that are also in the system summary. \n
                    * :darkblue[F1 score:] A measure of the harmonic mean of precision and recall. \n\n
                    Here are the results: 
                    """)
                    
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                    scores = scorer.score(st.session_state['answer'],
                                            st.session_state['vector_db_pandas_context'] )
                    st.write("Rouge Score 1: ",scores['rouge1'])
                    st.write("Rouge Score L: ",scores['rougeL'])
                    st.markdown( """<a style='display: block; text-align: center;' href="https://github.com/google-research/google-research/tree/master/rouge">Rouge Score</a>""",
                                unsafe_allow_html=True,
                                )

                # st.write("Here's the source from document: ")
                st.markdown("<h4 style='text-align: center; color: black;'>Here's the source from document:</h4>", unsafe_allow_html=True)
                st.dataframe(st.session_state['vector_db_pandas_matched_db'])
                st.markdown( """<a style='display: block; text-align: center;' href="https://en.wikipedia.org/wiki/Cosine_similarity">Score: Cosine Similarity</a>""",
                                unsafe_allow_html=True,
                                )

                st.markdown("<h4 style='text-align: center; color: black;'>Here's the summary of all sources:</h4>", unsafe_allow_html=True)
                prompt = f"""Write a concise summary of the following text delimited by triple backquotes.
                                Return your response in bullet points which covers the key points of the text.

                            ```{st.session_state['vector_db_pandas_context']}```

                            BULLET POINT SUMMARY: \n 
                            """
                st.write(get_text_generation(prompt=prompt))
            else: 
                st.write("Sorry we couldn't find anything around your query in the database, maybe try a different question?")
        

    else:
        # st.write(":red[Bada Bing Bada Boom].....Your need to :green[make your vector store] Groom . :brown[Upload document and Hit Processes]")
        st.markdown("<h5 style='text-align: center; color: darkred;'>Bada Bing Bada Boom, You need to make the vector store GROOOM...Upload document and Hit Processes</h5>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; color: black;'>Backend Engine Details: :</h4>", unsafe_allow_html=True)
    st.write("Document Processor: ",f":red[{st.session_state['processor_version']}]")
    st.write("Vector DB: ", f":red[{st.session_state['vector_db']}]")
    st.write("Chunk Size: ", st.session_state['chunk_size'])
    st.write("Top Results that are picked: ", st.session_state['top_sort_value'])
#Page Footer
st.write(html_comp.ft, unsafe_allow_html=True)

# st.markdown("![Alt Text](https://cdn.dribbble.com/userupload/3382003/file/original-7da4b54d8bb87c9a6f57dbd4a752e7c3.gif)")
# st.markdown(
#     f'<img src="data:image/gif;base64,{data_url}" alt="docy">',unsafe_allow_html=True,)
# file_ = open("./image/eyes.gif", "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# file_.close()
# st.markdown("<h5 style='text-align: center; color: darkblue;'>Contexual Enterprise Search</h5>", unsafe_allow_html=True)
# st.write(st.session_state['env_config']['gcp']['PROJECT_ID'])
# @st.cache_data
# def cache_vector_store(data):
#     vector_store = data.copy()
#     return vector_store

    # sample_bool_val = st.radio("Sample Vector Store?", (True, False))
    # st.session_state['sample_bool'] = sample_bool_val
    
    # if st.session_state['sample_bool']:
    #     sample_value = st.number_input("Sample Size:",value=10,min_value=1, max_value=10,step=1)
    #     st.session_state['sample_value'] = sample_value

    # if not st.session_state['process_doc']:
    #     if st.button("Process"):
    #         st.session_state['process_doc'] = True
    #         with st.spinner("Processing...this will take time..."):
    #             final_data = read_documents(user_docs,
    #                                         chunk_size_value=st.session_state['chunk_size'],
    #                                         sample=st.session_state['sample_bool'],
    #                                          sample_size=st.session_state['sample_value'])
    #             # vector_store = cache_vector_store(final_data)
    #             st.session_state['vector_store'] = final_data
    # else:
    #     st.write("Your vector store is already built. Here's the shape of it:")
    #     st.write(st.session_state['vector_store'].shape)
    
    # # if st.button("Relode/Reprocess files"):
    # #     st.session_state['process_doc'] = False

    # # if st.button("Reset Session"):
    # #     reset_session()