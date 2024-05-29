import streamlit as st
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage


####Streamlit web app for QnA from pdf embeddings######
####Global#############################################################
##Web Page meta
st.set_page_config(page_title="Document Q&A")
st.header("📑 Document Q&A Agent 🤖")

index_name = "doc-llm-index"
REDIS_URL = "redis://localhost:6379/0"
sesh_id = "1112"

#Initialize 
def _init():
    load_dotenv()
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))


####RAG###################################################################
##Retriever

def retrieve_vecstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'fetch_k': 10})
    
    ##Test 
    #query = "What is the requirement on cloud provider"
    #found_docs = docsearch.max_marginal_relevance_search(query, k=2, fetch_k=10)
    #for i, doc in enumerate(found_docs):
    #    print(f"Doc {i + 1}.", doc.metadata, "\n", doc.page_content, "\n")
    ##Test
    return retriever


##Generate
def query_llm():
    retriever = retrieve_vecstore()
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    ### Contextualize question ###
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

    ### Persist chat history in Redis
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


####Streamlit Web App#################################################################
def main():
    _init()
    retriever = retrieve_vecstore()
    session_history = RedisChatMessageHistory(sesh_id, url=REDIS_URL).messages
    #print("Session History", "\n", session_history, "\n")

    #Initialize session state with default message and llm chain
    if "messages" not in st.session_state.keys() or not session_history:
        st.session_state.messages = [{"role": "assistant", "content": "Please ask me any question about our company policies and guidelines"}]
        with st.chat_message("assistant"):
            st.write(st.session_state.messages[-1]["content"])

    #Display chat messages from history on app rerun
    #Test print chat history
    #st.write(session_history)
    for message in session_history:
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
                    config={"configurable": {"session_id": sesh_id}}
                )
                answer = response["answer"]
                #history = response["chat_history"]

                #Combine answer and document source
                #answer = str(output['answer']) + "\n\n" + "Source: " + output['context'][1].metadata['source']
                
                # Write answer to chat message container
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                #print("Chat History", "\n", history, "\n")


    ##Test
    #print("Chain with source", "\n Source:", output['context'][1].metadata['source'],"\n")


if __name__ == '__main__':
    #
    main()