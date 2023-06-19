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
    get_image_displayed(path='./image/bot2_small.png')
    st.markdown("<h1 style='text-align: center; color: blue;'>Customize your Docy</h1>", unsafe_allow_html=True)


    # Debug Mode
    debug_mode_choice = st.radio("Debug mode (keep it False)", (False,True),horizontal=True)
    st.session_state['debug_mode'] = debug_mode_choice

     # Demo Mode
    debug_mode_choice = st.radio("Demo mode (Capability Mode)", (False,True),horizontal=True)
    st.session_state['demo_mode'] = debug_mode_choice


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
        st.write("Coming Soon....Select other Vector DB - Pandas or Chroma")

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
                
            if not top_matched_df.empty: 
                st.session_state['vector_db_context'] =  context
                st.session_state['vector_db_matched_db'] = top_matched_df
                st.session_state['vector_db_source'] = source
                st.markdown("<h3 style='text-align: center; color: black;'>Extracted Answers and Insights</h3>", unsafe_allow_html=True)
                # st.write("Here's the answer from document: ")
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Grounded", "Focused","PaLMReImagine","PaLMBulletSummary","Evaluation","AskDocy"])
                
                with tab1:
                    st.write(":red[This uses Map Reduce Chains with Embedding]")
                    prompt = f""" Answer the question as precise as possible using the provided context. \n\n
                        Context: \n {st.session_state['vector_db_context']}?\n
                        Question: \n {question} \n
                        Answer:
                    """
                    answer = get_text_generation(prompt=prompt,temperature=0.0,max_output_tokens=1024)
                    st.session_state['answer'] = answer
                    st.write(st.session_state['answer'])

                with tab2:
                    st.write(":red[This uses focused Map Reduce Chains and finds answer in each relevant chunk]")

                    focused_answer,context, top_matched_df = get_focused_map_reduce_without_embedding(vector_db_choice = st.session_state['vector_db'],
                                     question = st.session_state['question'])
                    
                    st.session_state['focused_citation_df'] = top_matched_df
                    st.session_state['focused_answer'] = focused_answer
                    st.session_state['focused_answer_explainer'] = context
                    st.write(":green[Focused Answer]")
                    st.write(st.session_state['focused_answer'])
                    st.write(":green[Focused Raw]")
                    st.write(st.session_state['focused_answer_explainer'])
                   
                with tab3:
                    st.write("This will take the model response and write descriptive (essay/web article like) text which will include some data from its learning (from internet). The outputs will not be entirely based on the data and may hallucinate")
                    tab1, tab2 = st.tabs(["Grounded", "Focused"])
                    with tab1: 
                        prompt = f"""can you write a essay based on the following text. 
                        The essay should explain the points given in the {{text}}. 
                        Do not add anything on your own and only pick context from the {{text}}: \n text: \t {st.session_state['answer']}
                        """
                        rewritten_answer = get_text_generation(prompt=prompt,temperature=0.5,max_output_tokens=1024)
                        st.session_state['rewritten_answer']  = rewritten_answer
                        st.write(st.session_state['rewritten_answer'])
                    with tab2: 
                        prompt = f"""can you write a essay based on the following text. 
                        The essay should explain the points given in the {{text}}. 
                        Do not add anything on your own and only pick context from the {{text}}: \n text: \t {st.session_state['focused_answer_explainer']}
                        """
                        rewritten_answer = get_text_generation(prompt=prompt,temperature=0.5,max_output_tokens=1024)
                        st.session_state['rewritten_answer']  = rewritten_answer
                        st.write(st.session_state['rewritten_answer'])
                with tab4:
                    prompt = f"""Write a concise summary of the following text delimited by triple backquotes.
                                    Return your response in bullet points which covers the key points of the text.

                                ```{st.session_state['answer']}```

                                BULLET POINT SUMMARY: \n 
                                """
                    rewritten_bullet_answer = get_text_generation(prompt=prompt,temperature=0.5,max_output_tokens=1024)
                    st.session_state['rewritten_bullet_answer']  = rewritten_bullet_answer
                    st.write(st.session_state['rewritten_bullet_answer'])
                with tab5:
                    st.write(""":red[ROUGE] (Recall-Oriented Understudy for Gisting Evaluation) is a set of 
                    metrics for evaluating automatic summarization and machine translation software in natural language processing.""")
                    st.write(""":blue[The ROUGE-1:] It measures the overlap of unigrams between the system summary and the reference summary. """)
                    st.write(""":blue[The ROUGE-L:] It measures the longest common subsequence (LCS) between the system summary and the reference summary.""")
                    st.write(""":blue[Precision:] The fraction of the words in the system summary that are also in the reference summary.""")
                    st.write(""":blue[Recall:] The fraction of the words in the reference summary that are also in the system summary.""")
                    st.write(""":blue[F1 score:] A measure of the harmonic mean of precision and recall.""")
                    st.write(""":blue[Note:] We are using "Grounded" answer as a summary of "PaLMContext" to evaluate effectiveness of answers. Very rough approximation, not absolute.""")
                    st.write("""Here are the results: """)
                    
                    
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                    scores = scorer.score(st.session_state['answer'],
                                            st.session_state['vector_db_context'] )
                    st.write("Rouge Score 1: ",scores['rouge1'])
                    st.write("Rouge Score L: ",scores['rougeL'])
                    st.markdown( """<a style='display: block; text-align: center;' href="https://github.com/google-research/google-research/tree/master/rouge">Rouge Score</a>""",
                                unsafe_allow_html=True,
                                )
                    
                with tab6:
                    if st.button("Clear Chat"):
                        clear_chat()
                    if not st.session_state['chat_model']:
                        if st.button("Create your chat session"):
                            # with st.spinner('Loading documents using DocumentAI....'):
                            combined_input_context = st.session_state['vector_db_context'] 
                            final_context = f"""
                            Your name is Docy and you have been built on PaLM2. Your soul purpose is to be the best buddy to enterprise.
                            You do not respond with any other name. You are very funny when responding and people should feel they are talking to human. 
                            Your goal involves answering question based on the provided context as {{context:}}. You do not answer anything beside that. 
                            Your expertise is only on the context and anything outside of it should be responded politely saying that you can not respond to anything outside of context. \n
                            context: {combined_input_context}
                            """
                            chat_model = create_session(temperature = 0.1,
                                                                context= final_context
                                                                )
                            st.session_state['chat_model'] = chat_model
                            st.write("You are now connected to Docy built on PaLM 2 and ready to chat....")
                            st.write("here is the context: ",combined_input_context)
                    user_input = st.text_input('Your message to the Docy:', key='chat_widget', on_change=chat_input_submit)

                    if st.session_state.chat_input:
                        #call the vertex PaLM API and send the user input
                        with st.spinner('PaLM is working to respond back, wait.....'):
                            
                            # try:
                            bot_message = response(st.session_state['chat_model'], st.session_state.chat_input)
                        
                            #store the output
                            if len(st.session_state['past'])>0:
                                if st.session_state['past'][-1] != st.session_state.chat_input:
                                    st.session_state['past'].append(st.session_state.chat_input)
                                    st.session_state['generated'].append(bot_message)
                            else:
                                st.session_state['past'].append(st.session_state.chat_input)
                                st.session_state['generated'].append(bot_message)

                            # except AttributeError:
                            #     st.write("You have not created the chat session,click on 'Create your chat session'")

                    #display generated response 
                    if st.session_state['generated'] and st.session_state['past']:
                        for i in range(len(st.session_state["generated"])-1,-1,-1):
                            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='big-smile')
                            message(st.session_state["generated"][i], key=str(i), avatar_style='bottts')

                    if st.session_state['debug_mode']:
                        st.write("len of generated response: ",len(st.session_state["generated"]))
                        st.write(f'Last mssage to bot: {st.session_state.chat_input}')
                        st.write(st.session_state)

                # st.write("Here's the source from document: ")
                st.markdown("<h4 style='text-align: center; color: black;'>Sources & Citation</h4>", unsafe_allow_html=True)
                tab1, tab2 = st.tabs(["Grounded Citations", "Focused Citations"])
                with tab1: 
                    st.dataframe(st.session_state['vector_db_matched_db'])
                with tab2: 
                    st.dataframe(st.session_state['focused_citation_df'])
                st.markdown( """<a style='display: block; text-align: center;' href="https://en.wikipedia.org/wiki/Cosine_similarity">Score: Cosine Similarity</a>""",
                                unsafe_allow_html=True,
                                )

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
                    if not st.session_state['document_summary_mapreduce']:
                        with st.spinner('summarizing the document for you..beep boop..taking time....'):
                            summary_mapreduce = get_map_reduce_summary()
                            if summary_mapreduce:
                                st.session_state['document_summary_mapreduce'] = summary_mapreduce
                                st.write(st.session_state['document_summary_mapreduce'])
                    else:
                        st.write(st.session_state['document_summary_mapreduce'])

                with tab3: 
                    st.write("Here's verbatim of what we are sending to PaLM as context to answer your query: ")
                    st.write(st.session_state['vector_db_context'])
            else: 
                st.write("Sorry we couldn't find anything around your query in the database, maybe try a different question?")
        

    else:
        # st.write(":red[Bada Bing Bada Boom].....Your need to :green[make your vector store] Groom . :brown[Upload document and Hit Processes]")
        st.markdown("<h5 style='text-align: center; color: darkred;'>Bada Bing Bada Boom, You need to make the vector store GROOOM...Upload document and Hit Processes or use Demo Mode to see the capabilities of Docy.</h5>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; color: black;'>Backend Engine Details:</h4>", unsafe_allow_html=True)
    st.write("Document Processor: ",f":red[{st.session_state['processor_version']}]")
    st.write("Vector DB: ", f":red[{st.session_state['vector_db']}]")
    st.write("Chunk Size: ", st.session_state['chunk_size'])
    st.write("Top Results that are picked: ", st.session_state['top_sort_value'])
#Page Footer
st.write(html_comp.ft, unsafe_allow_html=True)