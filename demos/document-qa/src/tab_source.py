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
    # st.write(":red[This uses Map Reduce Chains with Embedding]")
    # given in {{context:}}. 
    #  Do not provide answers outside of the context. Do not add anything on your own when answering.
    prompt = f""" Answer the question as precise as possible using the provided context   \n\n
        Context: \n {context}?\n
        Question: \n {question} \n
        Answer:
    """
    answer = get_text_generation(prompt=prompt,temperature=0.0,max_output_tokens=1024)
    st.session_state['answer'] = answer
    st.write(st.session_state['answer'])
    if st.session_state['demo_mode'] != "Demo Mode":
        if st.button("Summarize the answer?"):
            st.write(":green[Here's the bullet point summary of the answer]")
            st.markdown(get_summary_for_answer_tab(answer = st.session_state['answer']))

def get_focused_answer_for_tab(question:str,vector_db_choice:str):
    # st.write(":red[This uses focused Map Reduce Chains and finds answer in each relevant chunk]")

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
        if st.button("Summarize the focused answer?"):
            st.write(":green[Here's the bullet point summary of the Focused answer]")
            st.markdown(get_summary_for_answer_tab(answer = st.session_state['focused_answer_explainer']))

def clear_chat() -> None:
    st.session_state['chat_model'] = ""
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['context'] = ""
    st.session_state['example'] = []
    st.session_state['temperature'] = []

def get_previous_question_answer_pairs(answer_pairs, question_pair):
    pairs = []
    for i in range(len(answer_pairs)):
        pairs.append(f"""previous_question: {question_pair[i]} \nprevious_answer:  {answer_pairs[i]}  \n""")
    return "\n".join(pairs)

def get_askdocy_for_tab():
    if st.button("Clear Chat"):
        clear_chat()
    # if st.button("Create your chat session"):
    user_input = st.text_input('Your message to the Docy:', key='chat_widget', on_change=chat_input_submit)
    if st.session_state.chat_input:
            context, top_matched_df, source = get_filter_context_from_vectordb(vector_db_choice = st.session_state['vector_db'],
                                        question = st.session_state.chat_input,
                                        sort_index_value =  2)
            if st.session_state['past'] and st.session_state['generated']:
                # previous_question: {st.session_state['past'][-5:]} \n
                # previous_answer:  {st.session_state['generated'][-5:]} \n
                if len(st.session_state['generated']) >= 5:
                    final_context = f"""
                    Your name is Docy and you have been built on PaLM2. You do not respond with any other name. 
                    Your goal involves answering question based on the provided context as {{context:}} and previous question and answers provided as as {{previous_question:}} and {{previous_answer}}. 
                    You do not answer anything beside that. If something is asked beyond that, say "Sorry, this is not part of my context and i can not answer it. 
                    You should also keep the context of the previous conversation and respond based on that by keeping the memory of it while responding to the current question. 
                    \n\n
                    {get_previous_question_answer_pairs(answer_pairs=st.session_state['generated'][-5:], 
                                                    question_pair=st.session_state['past'][-5:] )
                                                    }
                    context: {context} \n\n
                    """
                else:
                    # previous_question: {st.session_state['past'][-len(st.session_state['generated']):]} \n
                    # previous_answer:  {st.session_state['generated'][-len(st.session_state['generated']):]} \n
                    final_context = f"""
                    Your name is Docy and you have been built on PaLM2. You do not respond with any other name. 
                    Your goal involves answering question based on the provided context as {{context:}} and previous question and answers provided as as {{previous_question:}} and {{previous_answer}}. 
                    You do not answer anything beside that. If something is asked beyond that, say "Sorry, this is not part of my context and i can not answer it. 
                    You should also keep the context of the previous conversation and respond based on that by keeping the memory of it while responding to the current question. 
                    \n\n
                    {get_previous_question_answer_pairs(answer_pairs=st.session_state['generated'][-5:], 
                                                    question_pair=st.session_state['generated'][-5:] )
                                                    }
                                                    
                    context: {context} \n\n
                    """

            else:
                final_context = f"""
                Your name is Docy and you have been built on PaLM2.
                You do not respond with any other name. 
                Your goal involves answering question based on the provided context as {{context:}}. You do not answer anything beside that. 
                You should also keep the context of the previous conversation and respond based on that by keeping the memory of it while responding to the current question. 
                its available as {{previous_question:}} and {{previous_answer}} \n\n
                previous_question:  \n
                previous_answer:  \n
                context: {context} \n\n
                """

            # st.write("heres the context going to bot: ",final_context)
            chat_model = create_session(temperature = 0.1,
                                                context= final_context
                                                )
            st.session_state['chat_model'] = chat_model
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
                st.write("len of past response: ",len(st.session_state["past"]))    

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
            You should also keep the context of the previous conversation and respond based on that by keeping the memory of it while responding to the current question. 
            its available as {{previous_question:}} and {{previous_answer}} \n
            context: {combined_input_context} \n\n
            previous_question: {st.session_state['past'][-1]}
            previous_answer:  {st.session_state['generated'][-1]}
            """
            st.write("heres the context going to bot: ",final_context)
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


def get_document_link_for_demo_documents(document_choice):
    if document_choice == "HR Policy":
            url = "https://hr.karvy.com/HRpolicies/HR_Policy_Manual_KFSLnew.pdf"
            st.markdown(f"""<a style='display: block; text-align: center; color: red;' href={url}>View Document</a>""",
                                    unsafe_allow_html=True,
                                    )
    elif document_choice == "Rent Agreement":
        url = "https://assets1.cleartax-cdn.com/cleartax/images/1655708276_rentalagreementsampleandallyouneedtoknow.pdf"
        st.markdown(f"""<a style='display: block; text-align: center; color: red;' href={url}>View Document</a>""",
                                unsafe_allow_html=True,
                                )
    elif document_choice == "Health Insurance Policy":
        url = "https://nationalinsurance.nic.co.in/sites/default/files/1.%20Policy-%20NMP.pdf"
        st.markdown(f"""<a style='display: block; text-align: center; color: red;' href={url}>View Document</a>""",
                                unsafe_allow_html=True,
                                )
    
    elif document_choice == "Quarterly Earnings Report":
        url = "hhttps://abc.xyz/investor/static/pdf/2023Q1_alphabet_earnings_release.pdf"
        st.markdown(f"""<a style='display: block; text-align: center; color: red;' href={url}>View Document</a>""",
                                unsafe_allow_html=True,
                                )
    elif document_choice == "Bylaw":
        url = "https://www.imf.org/external/pubs/ft/bl/pdf/by-laws.pdf"
        st.markdown(f"""<a style='display: block; text-align: center; color: red;' href={url}>View Document</a>""",
                                unsafe_allow_html=True,
                                )


def get_sample_question_for_demo_documents(document_choice):
    if document_choice == "HR Policy":
            st.write("1. What are the interviewing guidelines?")
            st.write("2. What documents are required at the time of joining?")
            st.write("3. What is the cellphone usage policy? What are the limits for monthly reimbursement by level?")
            st.write("4. What are the conditions in which Gratuity is not paid. Please list them as individual points.")
            st.write("5. When is performance appraisal done in the company? What are the different ratings for any KRA?")
    elif document_choice == "Rent Agreement":
        st.write("1. What if there is a dispute in this agreement?")
        st.write("2. What is the notice period needed if one has to terminate this agreement?")
        st.write("3. If minor repairs need to be done, who is responsible for it? And what about major repairs?")
        st.write("4. Can the owner use this property for any purpose other than residing there? What is not allowed?")
        st.write("5. Under what conditions can the owner visit in person?")

    elif document_choice == "Health Insurance Policy":
        st.write("1. What are the systems of medicine supported?")
        st.write("2. What are the various discounts provided? Can you provide more information on the Online payment discount?")
        st.write("3. What are the contact details for the insurance person in Karnataka?")
        st.write("4. What are the time limits for submission of the claim documents?")
        st.write("5. What operations are included in the area of tonsils and adenoids?")
    
    elif document_choice == "Quarterly Earnings Report":
        st.write("1. What was the cost of reduction in workforce?")
        st.write("2. What is the total workforce in March 2023? How does it compare to that in March 2022? What does it mean in terms of percentage increase in employees?")
        st.write("3. Operating income loss is being reported for the Quarters ending 2022 and 2023. What are the groups under which it's reported? How did Cloud do in terms of business?")
        st.write("4. What was the total income from operations for quarter ending in March 31, 2023?")
        st.write("5. What is the total revenue for the quarter ending March 31, 2023? How much of a percentage increase or decrease over the same period last year?")
    elif document_choice == "Bylaw":
        st.write("1. What are the laws around proxy voting?")
        st.write("2. What happens when there is a vacancy for a Director?")
        st.write("3. What is the working language of the fund?")
        st.write("4. Who has certified this document?")
        st.write("5. What is section O-7?")
