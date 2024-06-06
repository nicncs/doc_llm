import streamlit as st
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os

st.title('🦜🔗 Quickstart App')

def generate_response(input_text):
    load_dotenv()
    llm = OpenAI(temperature=0.7, openai_api_key=os.getenv('OPENAI_API_KEY'))
    st.info(llm(input_text))

with st.form('my_form'):
    text = st.text_area('Enter text:', 'Ask me anything')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)