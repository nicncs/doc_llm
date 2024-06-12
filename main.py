import streamlit as st
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os
import redis
import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


####Streamlit web app for QnA from pdf embeddings######
####Global#############################################################
##Web Page meta
load_dotenv()

st.set_page_config(
    page_title="Document Q&A Assistant",
    layout="centered",
)
st.title("📑 Document Q&A Agent 🤖")
index_name = "esg-index" #Pinecone index name

#local redis
#REDIS_URL = "redis://localhost:6379"
#Docker redis
#REDIS_URL = "redis://redis:6379"
#r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

#Elasticache redis
REDIS_URL = os.getenv('REDIS_URL')
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
keys = r.keys()
# Get all values associated with the keys
#for key in keys:
#    print("\nRedis Dump Keys:", key)


####RAG###################################################################
##Retriever
def retrieve_vecstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 10})
    
        
    ##Test vector store result
    #query = "What is the requirement on cloud provider"
    #found_docs = docsearch.max_marginal_relevance_search(query, k=2, fetch_k=10)
    #for i, doc in enumerate(found_docs):
    #    print(f"Doc {i + 1}.", doc.metadata, "\n", doc.page_content, "\n")
    ##Test
    return retriever


##Generate LLM response
def query_llm():
    retriever = retrieve_vecstore()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    ### Contextualize question with chat history ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    #Prompt template
    qa_system_prompt = """You are an assistant for question-answering tasks and an expert in the policy documents of the company. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
    Answer the question based only on the following document context. \
    Cite which document this is from and the section number if applicable. \
    
    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Persist chat history in Redis by session_id and key_prefix ###
    def get_message_history(session_id: str, key_prefix: str = st.session_state["username"]) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(session_id, key_prefix=key_prefix, url=REDIS_URL)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

########## Other functions###################
#Redis functions
def list_session_ids(REDIS_URL, user_session):
    #get session list for current user
    session_list = r.get(user_session)
    if session_list:
        session_list = session_list.split(",")
        st.session_state["current_chat_index"] = len(session_list) - 1
    else:
        session_list = []
    return session_list

#Create chat session, generate new session ID based on datetime
def create_chat(user_session):
    session_id = "chat from " + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    # append session_id to session_list
    session_list = list_session_ids(REDIS_URL, user_session)
    session_list.append(session_id)
    #Update current user's session_list to redis
    r.set(user_session, ",".join(session_list))
    #Test
    #print("Session ID after create chat", session_list)
    #set st.session_state["current_chat_index"] to latest session
    st.session_state["current_chat_index"] = len(session_list) - 1
    st.session_state["current_session"] = session_id #set current session key for redis
    st.session_state.messages = [] #new conversation state
    st.rerun()


#Clear all chat session from redis with session_list key
@st.experimental_dialog("Confirm clear history?")
def delete_chat(user_session):
    # Delete session_List from redis
    if st.button("Delete"):
        with st.warning("Clearing chat history..."):
            #All chat history for user
            #Redis remove list of current user sessions and history
            r.delete(user_session, *r.keys(st.session_state["username"] + "*"))
            #r.flushdb()
            st.session_state["current_chat_index"] = 0
            st.session_state.messages = []
            st.rerun()

####Streamlit Web App#################################################################
def dashboard():
    #unique key for current user
    user_session = st.session_state["username"] + "session_list"
    retriever = retrieve_vecstore()
    session_list = list_session_ids(REDIS_URL, user_session)

    #Debug redis storage
    #print("\nCurrent user:", st.session_state["username"])
    #print("\nCurrent user_session:", session_list)

    #Buttons to create and delete conversations
    with st.sidebar:
        icon_text = f"""
        <div class="icon-text-container">
            <span style='font-size: 24px;'>Conversations</span>
        </div>
        """
        st.markdown(
            icon_text,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        create_chat_button = c1.button(
            "New", use_container_width=True, key="create_chat_button"
        )
        if create_chat_button:
            create_chat(user_session)

        delete_chat_button = c2.button(
            "Clear History", use_container_width=True, key="delete_chat_button"
        )
        if delete_chat_button:
            delete_chat(user_session)

    #Sidebar for conversation history
    with st.sidebar:
        chat_container = st.container()
        with chat_container:
            #create new chat after clearing history
            if not session_list:
                create_chat(user_session)
            current_chat = st.radio(
                label="conversations",
                format_func=lambda x: x.split("_")[0] if "_" in x else x,
                #List out available session_list as options
                options=session_list,
                label_visibility="collapsed",
                index=st.session_state["current_chat_index"],
                key="current_chat",
                #Call back on session change
                #on_change=session_callback()
            )
            #Current session key
            st.session_state["current_session"] = current_chat
        st.write("---")
    
    #Display current chat session title and history
    session_container = st.container(border=True)
    with session_container:
        st.write("Currently showing ", st.session_state["current_session"])
        #Get chat history from redis for current chat session if session_list is not empty
        if session_list:
            st.session_state.messages = RedisChatMessageHistory(st.session_state["current_session"], key_prefix=st.session_state["username"], url=REDIS_URL).messages
            #st.session_state.messages = r.get(str(st.session_state["current_session"]))
            #st.write("Chat History from Redis", "\n", st.session_state.messages, "\n")


    #List out the files in ./file_list.txt and add into string
    file_list = open("./file_list.txt", "r").read()
    with st.sidebar:
        st.write("### Available Documents for Q&A:")
        st.write(file_list)
    
    #Initialize new session state with default welcome message
    if "messages" not in st.session_state.keys() or not st.session_state.messages:
        st.session_state.messages.append(AIMessage(content="""Please ask me any question about our company policies and guidelines."""))


    #Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    
    #Chat input for user that will be prompt for llm
    if question := st.chat_input("Type your question"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(question)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=question))
        #st.session_state.messages.append({"role": "user", "content": question})
    
    #print("Chat History", st.session_state.messages)
    #Generate a new LLM response if last message is not from assistant
    if not isinstance(st.session_state.messages[-1], AIMessage):
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                source = query_llm()
                response = source.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": st.session_state["current_session"], "key_prefix": st.session_state["username"]}}
                )
                answer = response["answer"]
                #answer = "This is a test answer for " + st.session_state["current_session"] + " for the question " + question
                
                #Combine answer and document source
                #answer = str(response['answer']) + "\n\n" + "Source: " + response['context'][1].metadata['source']
                
                # Write answer to chat message container
                st.write(answer)
                #st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.messages.append(AIMessage(content=answer))
                #print(st.session_state.messages)


#Main function
def main():
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    if "initial_settings" not in st.session_state:
        with st.form("username_form"):
            st.subheader("Please enter your username to retrieve chat history")
            username = st.text_input("Username", value="demo-user")
            submitted = st.form_submit_button("Submit")
            print("Username: ", username, submitted, "\n")
            if submitted:
                st.session_state["username"] = username + ":"
                #Session state keys
                #Initialize 
                st.session_state["current_chat_index"] = 0
                st.session_state["current_session"] = ""
                st.session_state["initial_settings"] = True
                st.rerun()
    else:
        dashboard()

if __name__ == '__main__':
    #
    main()
