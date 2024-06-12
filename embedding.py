from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import nltk
from langchain.text_splitter import NLTKTextSplitter

#####Py script to load pdf documents, chunking, embedding and storing into Pinecone vector DB###
####Global#############################################################
index_name = "esg-index"

nltk.download('punkt')

#Initialize 
#Generate a .txt file that will list out in bullet format the files inside the /docs directory
def list_files():
    files = os.listdir("../ESG-docs")
    files = [file for file in files if not file.startswith('.')]  # Ignore files starting with '.'
    files.sort()  # Sort the files alphabetically
    with open("file_list.txt", "w") as f:
        for file in files:
            f.write(f"- {file}\n")
        
def _init():
    load_dotenv()
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    #Create index if not exists
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
    list_files()
  
####Indexing#############################################################


#PDF Doc Loader
def load_documents():
    loader = PyPDFDirectoryLoader("../ESG-docs")
    docs = loader.load()

    #print(docs[0].metadata['source'])
    #print(docs[0].page_content[:1000])
    #for i in range(len(docs)): print(docs[i].metadata['source'])

    return docs

#Text Splitter
def split_documents(documents):
    #Use recursive splitter
    #text_splitter = RecursiveCharacterTextSplitter(
    #chunk_size=1000, chunk_overlap=20, add_start_index=True)
    #chunks = text_splitter.split_documents(documents)

    #Use NLTK splitter
    text_splitter = NLTKTextSplitter()
    chunks = text_splitter.split_documents(documents)
    #Test
    #print(len(chunks))
    #print(chunks[1].metadata)
    #for i in range(len(chunks)): print(chunks[i].metadata)
    #Output chunks to file chunks.txt
    with open("chunks.txt", "w") as f:
        for chunk in chunks:
            f.write(f"Chunk Source: {chunk.metadata['source']}\n")
            f.write("Source" + chunk.page_content)
            f.write("\n\n")
    return chunks
  

#Embedding and Populate Vector Store
def embeddings_on_vecstore(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    #docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    vectorstore.add_documents(docs)


####Execute#################################################################
_init()
documents = load_documents()
chunks = split_documents(documents)
embeddings_on_vecstore(chunks)