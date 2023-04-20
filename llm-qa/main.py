##### IMPORTAR LIBRERÍAS #####
import streamlit as st
from langchain import OpenAI, VectorDBQA, LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.llms import BaseLLM
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import GoogleDriveLoader

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

from langchain.chains.base import Chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.chains.question_answering import load_qa_chain

from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from serpapi import GoogleSearch
from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

#import magic
import os
import nltk
#import config
import inspect
import tiktoken
from getpass import getpass
from collections import deque
from typing import Dict, List, Optional, Any

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)




##### CHATBOT CON ACCESO A INTERNET #####
# Definimos las herramientas que utilizará nuestro chatbot, en este caso solo utilizará SerpAPIWrapper para realizar búsquedas en Google. 
Google_search = SerpAPIWrapper()
toolsSERP = [
    Tool(
        name = "Current Search",
        func=Google_search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

# Definimos el modelo de lenguaje natural que utilizará nuestro chatbot.
llmSERP=ChatOpenAI(temperature=0,
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model_name="gpt-3.5-turbo"
)

# Definimos la cadena de herramientas que utilizará nuestro chatbot. Los elementos clave son tools y AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#que le permite al chatbot mantener una conversación a la par que utiliza herramientas .
agent_chain = initialize_agent(toolsSERP, llmSERP, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=ConversationBufferMemory(memory_key="chat_history"))

# Utilizamos DirectoryLoader para cargar los documentos fuente sobre los que entrenaremos el chatbot, en este caso, los documentos brindan conocimiento sobre Morada Uno.
loader_m1 = DirectoryLoader('https://github.com/fernando-m1/ai-m1/tree/main/llm-qa/Demo_docs')

# Extraemos el texto de los archivos PDF que cargamos.
documents_m1 = loader_m1.load()

# Aplicar RecursiveCharacterTextSplitter para partir los textos completos en pequeños trozos de texto que faciliten el trabajo al chatbot.
text_splitterRC = RecursiveCharacterTextSplitter(
    chunk_size = 500,  
    chunk_overlap  = 50,
    length_function = len
)
textsRC_m1 = text_splitterRC.split_documents(documents_m1)

# Definitmos el modelo que utilizaremos para realizar los embeddings (conversión de texto a vectores numéricos)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

# Creamos el vectorstore que alojará los embeddings creados en el paso anterior.
vectorstore_m1 = Chroma.from_documents(textsRC_m1, embeddings, metadatas=[{"source": str(i)} for i in range(len(textsRC_m1))])

# Definimos el modelo de lenguaje que utilizará nuestro chatbot
llm = OpenAI(
    batch_size=5,
    temperature=0,
    openai_api_key=os.environ['OPENAI_API_KEY']
)

qa_m1 = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=vectorstore_m1.as_retriever(), return_source_documents=False)

from langchain.agents.react.base import DocstoreExplorer

tools_m1 = [
    Tool(
        name="Morada Uno System",
        func=qa_m1.run,
        description="useful for when you need to look up information about Morada Uno"
    ),
    Tool(
        name="Google Search",
        func=Google_search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    )
]

memory=ConversationBufferMemory(return_messages=True, memory_key="chat_history")
react_chain = initialize_agent(tools_m1, llmSERP, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=False, memory=memory)



##### CÓDIGO PARA CARGAR STREAMLIT DE INICIO #####
st.set_page_config(layout="wide", page_title="Chatbot M1", page_icon="🤖")
st.title("Chatbot M1")
st.header("Chatbot con acceso a internet")

st.markdown("### **Escribe tus preguntas a Assistant, tu asesor de Morada Uno.**")
st.markdown("#### Tu preguntas:")
def get_text():
  input_text = st.text_area(label="", placeholder="Escribe aquí tus preguntas...", key="question_input")
  return input_text

question_input = get_text()

st.markdown("#### Assistant responde:")
if question_input:
  assistant_response = react_chain.run(input=question_input)
  st.write(assistant_response)
