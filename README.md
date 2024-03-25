# Airline Reviews Analysis App

## How to setup
1. clone this repository in your local machine using the following command
    `git clone https://github.com/c-viswanath/Airline-Reviews-Chatbot`
2. Create and start a new environment called `venv`
3. Install all the requirements from `requirements.txt`
4. Run `streamlit run app1.py` to run the app. (It will take some time for the first time to load the app because it has to generate the vectors, from the seconf time unless ou have made any changes to the vector file the speed will be much faster)    

## Description

This file contains the Streamlit app code for the Airline Reviews Analysis app. The app utilizes various natural language processing (NLP) techniques and tools, including GPT-3, to provide answers to user questions based on provided context.

**Import Necessary Libraries**

```python
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
```

This step imports all the necessary libraries and modules required for building the Streamlit app and performing natural language processing (NLP) tasks. These libraries include `Streamlit (streamlit)`, `FAISS (FAISS)`, `OpenAI embeddings (OpenAIEmbeddings)`, `ChatPromptTemplate (ChatPromptTemplate)`, `ChatOpenAI (ChatOpenAI)`, `Pandas (pd)`, and other modules for processing and handling data.

**Define Document Class**
```python
class Document:
    def __init__(self, page_content, metadata: Optional[dict] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}
```
This step defines a simple Python class named Document, which is used for storing the page content and metadata associated with each document. It has two attributes: `page_content` for storing the text content of the document and `metadata` for storing additional information if provided.

**Initialize ChatOpenAI and Prompt Template**
```python
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

```
This step initializes the `ChatOpenAI` instance to interact with the OpenAI API for language modeling tasks. It also defines a `ChatPromptTemplate` to specify the template for prompting questions based on provided context. The template includes placeholders for context and input question.

**Initialize OpenAI Embeddings and Check FAISS Index File**
```python
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

index_file_path = "data/faiss_index"
if os.path.exists(index_file_path):
    new_db = FAISS.load_local(index_file_path, embeddings, allow_dangerous_deserialization=True)
else:
    df = pd.read_csv('data/concatenated_data.csv')
    documents = [Document(text) for text in df['concatenated']]
    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local(index_file_path)
    new_db = FAISS.load_local(index_file_path, embeddings, allow_dangerous_deserialization=True)
```

In this step, the OpenAI embeddings are initialized using the provided API key. Then, the code checks if the FAISS index file exists. If it does, it loads the existing FAISS index from the file. Otherwise, it loads the data from a CSV file (`data/concatenated_data.csv`), creates a new **FAISS** index using the document embeddings, and saves the index to the specified file path.

 **Create Document and Retrieval Chains for Retrieval Augmented Generation(RAG) of Results**

```python
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = new_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```
This step creates document and retrieval chains for processing user queries. The `document_chain` is responsible for processing the user input and generating relevant documents, while the `retrieval_chain` combines the document retrieval process with language modeling to generate answers to user queries.

## Setting Up Streamlit Interface of Interactive Querying
```python
st.set_page_config(page_title='Airline Reviews Analysis', page_icon='✈️')
```

**Define App Layout and Input Form**
```python
st.title('Airline Reviews Analysis Using GPT-3')

with st.form('input_form'):
    input_prompt = st.text_input('Enter your question:')
    submit_button = st.form_submit_button(label='Get Answer')
```
This step defines the layout of the Streamlit app. It sets the app title and creates an input form for users to enter their questions. The form includes a text input field for entering the question and a submit button to trigger the processing of the question.
**Process Form Submission and Display Answer**
```python
if submit_button:
    response = retrieval_chain.invoke({"input": input_prompt})
    st.write('Answer:', response["answer"])
```
This step processes the form submission when the user clicks the submit button. It invokes the retrieval chain to generate an answer to the user's question based on the provided input prompt. The generated answer is then displayed on the app interface.

