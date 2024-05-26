## Conversational Q&A Chatbot

# to load env variables
from dotenv import load_dotenv
import os

# import inptut data
import PyPDF2
from PyPDF2 import PdfReader

# data pre-processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

# to create embeddings
from langchain.embeddings.openai import OpenAIEmbeddings

# connect openai llm 
import openai
from openai import OpenAI
#from langchain.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

# modelling
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

# created vector db
from langchain_community.vectorstores import FAISS

import streamlit as st
from streamlit_chat import message

import os
#os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Sidebar contents

# Inject custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 380px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title('👋 Greetings from PANGAEA AI assistant chatbot 👋')
    st.write('Specifically made for PANGAEA Data Curators')
    st.image(
            "https://www.pangaea.de/assets/social-icons/pangaea-share.png",
            width = 250
        )
    st.markdown('''
    Do you have queries regarding data curation in PANGAEA?
    
    Type in your questions here and click 'Enter'.
    Type of questions:
    1. Can I delete a published dataset with open access?
    2. What is the document about?
    ...
    
    Source
    - [PANGAEA wiki](https://wiki.pangaea.de/wiki/Intern:Versioning)
    
    This is an LLM-powered application built using OpenAI:
    - [OpenAI](https://platform.openai.com/account/limits)

    ''')
    
# create prompt:

prompt_template="""
Answer the question based on the context below, and if the question can't be answered based on the context, say "sorry, I do not have that information" 

Context: {context}

---

Question: {question}

Answer:
"""

pdf_docs = ['data/Intern:Versioning.pdf']

def prepare_docs(pdf_docs):
    docs = []
    metadata = []
    content = []

    for pdf in pdf_docs:

        pdf_reader = PyPDF2.PdfReader(pdf)
        for index, text in enumerate(pdf_reader.pages):
            doc_page = {'title': pdf + " page " + str(index + 1),
                        'content': pdf_reader.pages[index].extract_text()}
            docs.append(doc_page)
    for doc in docs:
        content.append(doc["content"])
        metadata.append({
            "title": doc["title"]
        })
    print("Content and metadata are extracted from the documents")
    return content, metadata

content, metadata = prepare_docs(pdf_docs)

def get_text_chunks(content, metadata):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=150,
    )
    split_docs = text_splitter.create_documents(content, metadatas=metadata)
    print(f"Split documents into {len(split_docs)} passages")
    return split_docs

split_docs = get_text_chunks(content, metadata)

def ingest_into_vectordb(split_docs):
    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    print('Vector DB created')
    return db

vectordb=ingest_into_vectordb(split_docs)

def get_conversation_chain(vectordb):
    #llm_openai = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0)
    llm_openai = ChatOpenAI(model_name = "gpt-3.5-turbo-0125", temperature=0)
    retriever = vectordb.as_retriever()
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(prompt_template)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = (ConversationalRetrievalChain.from_llm
                          (llm=llm_openai,
                           retriever=retriever,
                           condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                           memory=memory,
                           return_source_documents=True))
    print("Conversational Chain created for the LLM using the vector store")
    return conversation_chain

conversation_chain=get_conversation_chain(vectordb)

#st.title("PANGAEA AI assistant ChatBot 👩‍💻🧑‍💻")
#st.subheader('Generated from _OpenAI_', divider='rainbow')

header = st.container()
header.title("PANGAEA AI assistant ChatBot 👩🏻‍💻🧑🏻‍💻")
#header.subheader('Generated from _OpenAI_', divider='rainbow')
header.subheader(divider='rainbow')
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

### Custom CSS for the sticky header
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;
        background-color: white;
        z-index: 999;
    }
    .fixed-header {
        border-bottom: 1px solid white;
    }
</style>
    """,
    unsafe_allow_html=True
)

def conversation_chat(query):
    result = conversation_chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hallo Data Curator! How may I assist you today?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hallo! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about PANGAEA data curation", key='input')
            submit_button = st.form_submit_button(label='Enter')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

# Initialize session state
initialize_session_state()

# Display chat history
display_chat_history()