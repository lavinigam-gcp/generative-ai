import streamlit as st
st.set_page_config(
    page_title="Question Answering using Vertex PaLM API",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/GoogleCloudPlatform/generative-ai',
        'Report a bug': "https://github.com/GoogleCloudPlatform/generative-ai/issues",
        'About': "# This app shows you how to use Vertex PaLM API on your custom documents"
    }
)
from src.session_states import *
import src.html_component as html_comp
from ruamel.yaml import YAML
from pathlib import Path


@st.cache_data()
def get_env_config():
    # st.write("Loading env.yaml")
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
import src.tab_source as tab_source

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
    get_image_displayed(path='./image/bot2_small.png')
    st.markdown("<h1 style='text-align: center; color: blue;'>Customize your Docy</h1>", unsafe_allow_html=True)


    # Debug Mode
    debug_mode_choice = st.radio("Debug mode (keep it False)", (False,True),horizontal=True)
    st.session_state['debug_mode'] = debug_mode_choice

     # Demo Mode
    demo_mode = st.radio("App mode:", ("Demo Mode","Auto Mode","Advanced Mode"))
    st.session_state['demo_mode'] = demo_mode

    if st.session_state['demo_mode'] == "Demo Mode":
        ## Load the predefined vector store from .temp folder 
        st.write("You have selected Demo Mode. We will use the predefined vector store.") 
        # st.session_state['vector_store_data_typedf'] = st.session_state['demo_mode_vector_store_data_typedf'].copy()
        if not st.session_state['demo_mode_vector_store_data_typedf'].empty:
            st.session_state['vector_db'] = "Demo Mode"
            st.session_state['top_sort_value'] = 5

    
    else:
        if st.session_state['demo_mode'] == "Auto Mode":
            # st.session_state['vector_store_flag'] = False
            st.write("You have selected Auto Mode. We will use the default settings.")
            st.session_state['processor_version'] = "OpenSource(PyPDF2)"
            st.session_state['vector_db'] = "Pandas"
            st.session_state['chunk_size'] = 2000
            st.session_state['top_sort_value'] = 3
            st.write("The default settings are:")
            st.write("Document Processor: ", st.session_state['processor_version'])
            st.write("Vector DB: ", st.session_state['vector_db'])
            st.write("Chunk Size: ", st.session_state['chunk_size'])
            st.write("Top N results: ", st.session_state['top_sort_value'])

        elif st.session_state['demo_mode'] == "Advanced Mode":
            # st.session_state['vector_store_flag'] = False
            # Select the processor version [OpenSource(PyPDF2), DocumentAI]
            processor_version = st.radio("Select the Document Processor", ( "OpenSource(PyPDF2)", "DocumentAI"),horizontal=True)
            st.session_state['processor_version'] = processor_version
            if st.session_state['processor_version'] == "DocumentAI":
                st.write("*As of now you can only upload PDF documents with limit of upto 14 pages. Multiple document supported.")
            elif st.session_state['processor_version'] == "OpenSource(PyPDF2)":
                st.write("*As of now you can only upload PDF documents. Multiple document supported. Using PyPDF2")
            # Select Vector DB [Pandas, Faiss, Chroma, MatchingEngine]
            vector_db = st.radio("Select the Vector DB", ( "Pandas", "Chroma"))
            st.session_state['vector_db'] = vector_db
            # Selecting the chunk size
            chunk_size_value = st.number_input("Chunk Size:",value=500,min_value=200, max_value=10000,step=200)
            st.session_state['chunk_size'] = chunk_size_value
            # Select Top N results
            top_match_result_value = st.number_input("How many result to pick?",value=3)
            st.session_state['top_sort_value'] = top_match_result_value

        if not st.session_state['file_upload_flag']:
            user_docs = st.file_uploader(
                    "Upload your documents. PDF Only", accept_multiple_files=True, type=['pdf'],
                    on_change = change_file_upload_flag)
            

        # Loading & Processing Documents as per the selected Method
        if not st.session_state['process_doc']:
            if st.button("Process Documents"):
                
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
                    # st.session_state['vector_store_data_typedf'].to_csv('./temp/vector_store_data_typedf_google10k.csv')


                # Summarizing the documents
                if not st.session_state['document_summary_mapreduce']:
                    with st.spinner('summarizing the document for you..beep boop..taking time....'):
                        summary_mapreduce = get_map_reduce_summary()
                        if summary_mapreduce:
                            st.session_state['document_summary_mapreduce'] = summary_mapreduce

                #Getting Nearest Neighbors to user Query
                # Happening inside container 
            
    # else:
    #     st.write("Your document is already loaded, processed and pushed to Vector DB. Ask away your question -->")

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

    if st.session_state['demo_mode'] == "Demo Mode":
        st.markdown( """<a style='display: block; text-align: center;' href="https://abc.xyz/investor/static/pdf/20230426_alphabet_10Q.pdf">Check the source of indexed document</a>""",
                                unsafe_allow_html=True,
                                )



    if (st.session_state['vector_store_flag'] and st.session_state['demo_mode'] != "Demo Mode")  or (st.session_state['vector_store_flag_demo'] and st.session_state['demo_mode'] == "Demo Mode"):
        
        question = st.text_input('What would you like to ask the documents?')
        st.session_state['question'] = question
        # get the custom relevant chunks from all the chunks in vector store.
        if question:
            if st.session_state['vector_db'] == "Pandas":
                context, top_matched_df, source = get_filter_context_from_vectordb(vector_db_choice = st.session_state['vector_db'],
                                     question = st.session_state['question'],
                                     sort_index_value =  st.session_state['top_sort_value'])
            elif st.session_state['vector_db'] == "Chroma":
                context, top_matched_df, source = get_filter_context_from_vectordb(vector_db_choice = st.session_state['vector_db'],
                                     question = st.session_state['question'],
                                     sort_index_value =  st.session_state['top_sort_value'])
            elif st.session_state['vector_db'] == "Demo Mode":
                context, top_matched_df, source = get_filter_context_from_vectordb(vector_db_choice = st.session_state['vector_db'],
                                     question = st.session_state['question'],
                                     sort_index_value =  st.session_state['top_sort_value'])
                
            if not top_matched_df.empty: 
                st.session_state['vector_db_context'] =  context
                st.session_state['vector_db_matched_db'] = top_matched_df
                st.session_state['vector_db_source'] = source
                
                #Extracted Answer and Insights Block: 
                st.markdown("<h3 style='text-align: center; color: black;'>Extracted Answers and Insights</h3>", unsafe_allow_html=True)
                # st.write("Here's the answer from document: ")
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Grounded", "Focused","PaLMReImagine","Evaluation","AskDocy"])
                with tab1:
                    tab_source.get_grounded_answer_for_tab(question=question, context=st.session_state['vector_db_context'])
                with tab2:
                    tab_source.get_focused_answer_for_tab(question=question, vector_db_choice = st.session_state['vector_db'])   
                with tab3:
                    tab_source.get_palmreimagine_answer_for_tab(answer=st.session_state['answer'])
                with tab4:
                    tab_source.get_evaluation_for_tab()
                with tab5:
                    tab_source.get_chat_for_tab()
                    

                #Sources & Citation Block:
                st.markdown("<h4 style='text-align: center; color: black;'>Sources & Citation</h4>", unsafe_allow_html=True)
                tab1, tab2 = st.tabs(["Grounded Citations", "Focused Citations"])
                with tab1: 
                    st.dataframe(st.session_state['vector_db_matched_db'])
                with tab2: 
                    st.dataframe(st.session_state['focused_citation_df'])
                st.markdown( """<a style='display: block; text-align: center;' href="https://en.wikipedia.org/wiki/Cosine_similarity">Score: Cosine Similarity</a>""",
                                unsafe_allow_html=True,
                                )
                
                #Summary Block: 
                st.markdown("<h4 style='text-align: center; color: black;'>Summary</h4>", unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["Source Summary", "Document Summary","PaLMContext" ])
                with tab1: 
                    prompt = f"""Write a concise summary of the following text delimited by triple backquotes.
                                    Return your response in bullet points which covers the key points of the text.

                                ```{st.session_state['vector_db_context']}```

                                BULLET POINT SUMMARY: \n 
                                """
                    st.write(get_text_generation(prompt=prompt))
                with tab2: 
                    st.write("Using Map Reduce Summary Chain Method:")
                    st.write("Currently we are using the first 10 chunks of the document to generate the summary.")
                    if st.session_state['document_summary_mapreduce']:
                       st.write(st.session_state['document_summary_mapreduce'])
                    else:
                        st.write("Summary of the document not available.")

                with tab3: 
                    st.write("Here's verbatim of what we are sending to PaLM as context to answer your query: ")
                    st.write(st.session_state['vector_db_context'])
            else: 
                st.write("Sorry we couldn't find anything around your query in the database, maybe try a different question?")
        

    else:
        # st.write(":red[Bada Bing Bada Boom].....Your need to :green[make your vector store] Groom . :brown[Upload document and Hit Processes]")
        st.markdown("<h5 style='text-align: center; color: darkred;'>Bada Bing Bada Boom, You need to make the vector store GROOOM...Upload document and Hit Processes or use Demo Mode to see the capabilities of Docy.</h5>", unsafe_allow_html=True)

    if st.session_state['demo_mode'] != "Demo Mode":
        st.markdown("<h4 style='text-align: center; color: black;'>Backend Engine Details:</h4>", unsafe_allow_html=True)
        st.write("Document Processor: ",f":red[{st.session_state['processor_version']}]")
        st.write("Vector DB: ", f":red[{st.session_state['vector_db']}]")
        st.write("Chunk Size: ", st.session_state['chunk_size'])
        st.write("Top Results that are picked: ", st.session_state['top_sort_value'])
#Page Footer
st.write(html_comp.ft, unsafe_allow_html=True)