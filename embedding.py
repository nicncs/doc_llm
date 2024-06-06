from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

#####Py script to load pdf documents, chunking, embedding and storing into Pinecone vector DB###
####Global#############################################################
index_name = "doc-llm-index"

#Initialize 
def _init():
    load_dotenv()
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 
  
####Indexing#############################################################

#PDF Doc Loader
def load_documents():
    loader = PyPDFDirectoryLoader("docs/")
    docs = loader.load()

    #generate .txt file that will list out in bullet format the filenames from ./docs directory
    with open('document_list.txt', 'w') as f:
        for item in loader.files:
            f.write("%s\n" % item)

    #print(docs[0].metadata['source'])
    #print(docs[0].page_content[:1000])
    #for i in range(len(docs)): print(docs[i].metadata['source'])

    return docs

#Text Splitter
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True)
    chunks = text_splitter.split_documents(documents)

    #Test
    #print(len(chunks))
    #print(chunks[1].metadata)
    #for i in range(len(chunks)): print(chunks[i].metadata)

    return chunks
  

#Embedding and Populate Vector Store
def embeddings_on_vecstore(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    vectorstore.add_documents(docs)

#Generate a json file consisting of bullet list the filenames from ./docs directory
def generate_json():



####Execute#################################################################
_init()
documents = load_documents()
chunks = split_documents(documents)
embeddings_on_vecstore(chunks)