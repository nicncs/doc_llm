# Project to create RAG application

An AI agent with a web interface that helps user search document content from a private document storage, and returns result of relevant content with reference to the document title.

## Features:

### MVP1:
1. Web app chat interface for Q&A based on the document contents
2. Document loader from pre-populated local folder
3. PDF documents reader 
5. Hallucination prevention - answer don’t know if not from any of the document content


### MVP2 (future): 
1. Session chat history to return to previous results
2. Load document from Google drive or company doc directory
3. Results will reference to document file name / title 
4. Return images


### Tools used:
- Streamlit
- Langchain
- Pinecone
- Python3.9

### Usage

1. Clone the repository to your local machine:

   ```bash
   git clone repo
   cd into working directory
   ```

2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

3. Embedding documents and update vector store:
   ```bash
   streamlit run embedding.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run rag.py
   ```

   