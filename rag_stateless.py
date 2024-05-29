import streamlit as st
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

####Streamlit web app for QnA from pdf embeddings######
####Global#############################################################
##Web Page meta
st.set_page_config(page_title="Document Q&A")
st.header("📑 Document Q&A Agent 🤖")

index_name = "doc-llm-index"

#Prompt template
template = """You are an assistant for question-answering tasks and an expert in the policy documents of the company. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Answer the question based only on the following document context: {context}
Question: {question}
Helpful Answer:"""


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

##Format docs if needed
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


##Generate
def query_llm(retriever):
    llm = ChatOpenAI(model="gpt-4-turbo")
    retriever = retrieve_vecstore()

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
    )

    ## Without source:
    #chain = setup_and_retrieval | prompt | llm | output_parser
    ## With source:
    rag_chain_from_docs = RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) | prompt | llm | output_parser
    chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs) 
    ###

    ##Test streaming answer
    #for answer in chain.stream("Summarize Appendix 10 of RMIT?"):
    #    print(answer, end="", flush=True)
    ##

    return chain_with_source


####Streamlit Web App#################################################################
def main():
    _init()
    retriever = retrieve_vecstore()
    #chain = query_llm(retriever)
    #Initialize session state with default message and llm chain
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Please ask me any question about our company policies and guidelines"}]

    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    
    #Chat input for user that will be prompt for llm
    if question := st.chat_input("Type your question"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.write(question)
            # print("User prompt entered:",prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})




    #Generate a new LLM response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain_with_source = query_llm(retriever)
                source = chain_with_source.stream(question)

                # Compile stream as it's being returned
                output = {}
                for chunk in source:
                    for key in chunk:
                        if key not in output:
                            output[key] = chunk[key]
                        else:
                            output[key] += chunk[key]

                #Combine answer and document source
                answer = str(output['answer']) + "\n\n" + "Source: " + output['context'][1].metadata['source']
                st.write(answer)

                # Write answer to chat message container
                message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(message)

            ##Test
            #print("Answer", "\n", output['answer'], "\n")
            #print("Chain with source", "\n Source:", output['context'][1].metadata['source'],"\n")


if __name__ == '__main__':
    #
    main()