import streamlit as st
from src.session_states import *
from src.utils import *
from rouge_score import rouge_scorer


def get_summary_for_answer_tab(answer:str):
    prompt = f"""Write a concise summary of the following text delimited by triple backquotes.
                Return your response in bullet points which covers the key points of the text.

            ```{answer}```

            BULLET POINT SUMMARY: \n 
            """
    rewritten_bullet_answer = get_text_generation(prompt=prompt,temperature=0.5,max_output_tokens=1024)
    st.session_state['rewritten_bullet_answer']  = rewritten_bullet_answer
    return st.session_state['rewritten_bullet_answer']

def get_palmreimagine_answer_for_tab(answer:str):
    st.write("This will take the model response and write descriptive (essay/web article like) text which will include some data from its learning (from internet). The outputs will not be entirely based on the data and may hallucinate")
    tab1, tab2 = st.tabs(["Grounded", "Focused"])
    with tab1:
        if st.button("Create for Grounded!"):
            prompt = f"""can you write a essay based on the following text. 
            The essay should explain the points given in the {{text}}. 
            Do not add anything on your own and only pick context from the {{text}}: \n text: \t {answer}
            """
            rewritten_answer = get_text_generation(prompt=prompt,temperature=0.5,max_output_tokens=1024)
            st.session_state['rewritten_answer']  = rewritten_answer
            st.write(st.session_state['rewritten_answer'])
    with tab2:
        if st.button("Create for Focused!"):
            prompt = f"""can you write a essay based on the following text. 
            The essay should explain the points given in the {{text}}. 
            Do not add anything on your own and only pick context from the {{text}}: \n text: \t {answer}
            """
            rewritten_answer = get_text_generation(prompt=prompt,temperature=0.5,max_output_tokens=1024)
            st.session_state['rewritten_answer']  = rewritten_answer
            st.write(st.session_state['rewritten_answer'])

def get_grounded_answer_for_tab(question:str,context:str):
    st.write(":red[This uses Map Reduce Chains with Embedding]")
    prompt = f""" Answer the question as precise as possible using the provided context. \n\n
        Context: \n {context}?\n
        Question: \n {question} \n
        Answer:
    """
    answer = get_text_generation(prompt=prompt,temperature=0.0,max_output_tokens=1024)
    st.session_state['answer'] = answer
    st.write(st.session_state['answer'])
    if st.button("Summarize the answer?"):
        st.write(":green[Here's the bullet point summary of the answer]")
        st.markdown(get_summary_for_answer_tab(answer = st.session_state['answer']))

def get_focused_answer_for_tab(question:str,vector_db_choice:str):
    st.write(":red[This uses focused Map Reduce Chains and finds answer in each relevant chunk]")

    if st.button("Get more focused answer"):

        focused_answer,context, top_matched_df = get_focused_map_reduce_without_embedding(question = st.session_state['question'],
                                                                                        vector_db_choice = vector_db_choice,
                                                                                        )
        
        st.session_state['focused_citation_df'] = top_matched_df
        st.session_state['focused_answer'] = focused_answer
        st.session_state['focused_answer_explainer'] = context
        st.write(":green[Focused Answer]")
        st.write(st.session_state['focused_answer'])
        st.write(":green[Focused Answer Explainer]")
        st.write(st.session_state['focused_answer_explainer'])
        if st.button("Summarize the answer?"):
            st.write(":green[Here's the bullet point summary of the Focused answer]")
            st.markdown(get_summary_for_answer_tab(answer = st.session_state['focused_answer_explainer']))


def get_chat_for_tab():
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
    if st.session_state['chat_model']:
        st.write("You are now connected to Docy built on PaLM 2 and ready to chat....")
        st.write("Note: You can only chat based on the question you have asked and context around it. You can not ask anything outside of the question and its context.")
    # st.write("here is the context: ",final_context)
    user_input = st.text_input('Your message to the Docy:', key='chat_widget', on_change=chat_input_submit)

    if st.session_state.chat_input:
        #call the vertex PaLM API and send the user input
        with st.spinner('PaLM is working to respond back, wait.....'):
            
            try:
                bot_message = response(st.session_state['chat_model'], st.session_state.chat_input)
      
                #store the output
                if len(st.session_state['past'])>0:
                    if st.session_state['past'][-1] != st.session_state.chat_input:
                        st.session_state['past'].append(st.session_state.chat_input)
                        st.session_state['generated'].append(bot_message)
                else:
                    st.session_state['past'].append(st.session_state.chat_input)
                    st.session_state['generated'].append(bot_message)

            except AttributeError:
                st.write("You have not created the chat session,click on 'Create your chat session'")

    #display generated response 
    if st.session_state['generated'] and st.session_state['past']:
        for i in range(len(st.session_state["generated"])-1,-1,-1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='big-smile')
            message(st.session_state["generated"][i], key=str(i), avatar_style='bottts')

    if st.session_state['debug_mode']:
        st.write("len of generated response: ",len(st.session_state["generated"]))
        st.write(f'Last mssage to bot: {st.session_state.chat_input}')
        st.write(st.session_state)

def get_evaluation_for_tab():
    if st.button("Evaluate the results!"):
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
