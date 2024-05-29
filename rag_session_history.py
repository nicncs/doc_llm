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

st.set_page_config(
    page_title="Document Q&A Assistant",
    layout="wide",
)
st.header("📑 Document Q&A Agent 🤖")
index_name = "doc-llm-index" #Pinecone index name
REDIS_URL = "redis://localhost:6379/0" #Docker redis image
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)


#Initialize 
if "initial_settings" not in st.session_state:
    load_dotenv()
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    #Session state keys
    st.session_state["current_chat_index"] = 0
    st.session_state["current_session"] = ""
    st.session_state["initial_settings"] = True

####RAG###################################################################
##Retriever
def retrieve_vecstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'fetch_k': 10})
    
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
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

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

    ### Persist chat history in Redis by session_id
    def get_message_history(session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(session_id, url=REDIS_URL)

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
def list_session_ids(REDIS_URL):
    #Get session_list from redis
    #r = redis.Redis.from_url(REDIS_URL)
    session_list = r.get("session_list")
    if session_list:
        session_list = session_list.split(",")
    else:
        session_list = []
    return session_list

#Create chat session, generate new session ID based on datetime
def create_chat():
    session_id = "chat from " + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    # append session_id to session_list
    session_list = list_session_ids(REDIS_URL)
    session_list.append(session_id)
    #Update session_list to redis
    #r = redis.Redis.from_url(REDIS_URL)
    r.set("session_list", ",".join(session_list))
    #Test
    #print("Session ID after create chat", session_list)
    #set st.session_state["current_chat_index"] to latest session
    st.session_state["current_chat_index"] = len(session_list) - 1
    st.session_state["current_session"] = session_id #set current session key for redis
    st.session_state.messages = [] #new conversation state
    st.rerun()


#Clear all chat session from redis with session_list key
@st.experimental_dialog("Confirm clear history?")
def delete_chat():
    # Delete session_List from redis
    #r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    if st.button("Delete"):
        with st.warning("Clearing chat history..."):
            #r.delete("session_list") #only remove list of sessions, not chat history
            r.flushdb()
            st.session_state["current_chat_index"] = 0
            st.session_state.messages = []
            st.rerun()

###
#def session_callback():
    #Clear chat session state
    #st.session_state.messages = []
    

####Streamlit Web App#################################################################
def main():
    retriever = retrieve_vecstore()
    session_list = list_session_ids(REDIS_URL)

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
            create_chat()

        delete_chat_button = c2.button(
            "Clear History", use_container_width=True, key="delete_chat_button"
        )
        if delete_chat_button:
            delete_chat()

    #Sidebar for conversation history
    with st.sidebar:
        chat_container = st.container()
        with chat_container:
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
            st.session_state.messages = RedisChatMessageHistory("1111", url=REDIS_URL).messages
            #st.session_state.messages = r.get(str(st.session_state["current_session"]))
            st.write("Chat History from Redis", "\n", st.session_state.messages, "\n")


    #Initialize session state with default message and llm chain
    if "messages" not in st.session_state.keys() or not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "Please ask me any question about our company policies and guidelines"}]
        with st.chat_message("assistant"):
            st.write(st.session_state.messages[-1]["content"])

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
        st.session_state.messages.append({"role": "user", "content": question})
    
    
    #Generate a new LLM response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                source = query_llm()
                response = source.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "1111"}}
                )
                answer = response["answer"]
                #answer = "This is a test answer for " + st.session_state["current_session"] + " for the question " + question
                
                #Combine answer and document source
                #answer = str(output['answer']) + "\n\n" + "Source: " + output['context'][1].metadata['source']
                
                # Write answer to chat message container
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                #Load messages to redis using RedisChatMessageHistory.aadd_messages()
                #RedisChatMessageHistory("1111", url=REDIS_URL).aadd_messages(st.session_state.messages) 
                print(st.session_state.messages)



    ##Test
    #print("Chain with source", "\n Source:", output['context'][1].metadata['source'],"\n")


if __name__ == '__main__':
    #
    main()
