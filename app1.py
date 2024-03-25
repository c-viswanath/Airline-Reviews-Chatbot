# Description: This file contains the Streamlit app code for the Airline Reviews Analysis app.
#IMPORT NECESSARY LIBRARIES
import streamlit as st
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from typing import Optional

# Define the Document class for storing the page content and metadata
class Document:
    def __init__(self, page_content, metadata: Optional[dict] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Initialize ChatOpenAI
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

# Check if the FAISS index file exists
index_file_path = "data/faiss_index"
if os.path.exists(index_file_path):
    # Load the existing FAISS index
    new_db = FAISS.load_local(index_file_path, embeddings, allow_dangerous_deserialization=True)
else:
    # Load the data and create a new FAISS index
    df = pd.read_csv('data/concatenated_data.csv')
    documents = [Document(text) for text in df['concatenated']]
    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local(index_file_path)
    new_db = FAISS.load_local(index_file_path, embeddings, allow_dangerous_deserialization=True)

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = new_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Set Streamlit app title and layout
st.set_page_config(page_title='Airline Reviews Analysis', page_icon='✈️')

# Define the app layout
st.title('Airline Reviews Analysis Using GPT-3')

# Create a form for input submission
with st.form('input_form'):
    # Add a text input field for the question
    input_prompt = st.text_input('Enter your question:')
    
    # Add a submit button
    submit_button = st.form_submit_button(label='Get Answer')

# Process the form submission
if submit_button:
    # Invoke retrieval chain to get the answer
    response = retrieval_chain.invoke({"input": input_prompt})
    
    # Display the answer
    st.write('Answer:', response["answer"])
