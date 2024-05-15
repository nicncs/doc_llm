# Project to create RAG application

An AI agent with a web interface that helps user search document content from a private document storage, and returns result of relevant content with reference to the document title.

## Features:

### MVP1:
1. Web app
2. Document loader from local folder
3. PDF documents reader 
4. Chat interface for Q&A based on the document contents

### MVP2: 
1. Session chat history to return to previous results
2. Hallucination prevention - answer don’t know if not from any of the document content
3. Load document from Google drive or company doc directory
4. Results will reference to document file name / title 
5. Return images


### Tools used:
- Streamlit
- Langchain
- Pinecone
- Python3.9