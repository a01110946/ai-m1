import streamlit as st

from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler 
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain, create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, GCSDirectoryLoader, JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.schema import Document, SystemMessage, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool, Tool
from langchain.vectorstores.faiss import FAISS
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

from typing import Any, Iterator, List
import requests
from PIL import Image
from io import BytesIO

# Streamlit page config
st.set_page_config(page_title="Hannah Bonvoy", page_icon="🤖")

# Page title
st.title("Hannah Bonvoy 🤖")

# Load M1 image
response_img = requests.get("https://raw.githubusercontent.com/a01110946/hannah_bonvoy/d2732e956da97213cdcb040f853283697bf67fd4/llm-qa/assets/images/FPbS_logo.png")

# Ensure that the request was successful and the content is indeed an image
if response_img.status_code == 200:
    img = Image.open(BytesIO(response_img.content))
else:
    raise ValueError("Could not retrieve the image - HTTP status code: {}".format(response_img.status_code))


# Define functions
def text_splitter_func():
  return RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\n\n", "####","\n","."],
    chunk_size=1500,
    chunk_overlap=500,
  )

def gcs_loader(bucket, project_name, prefix, text_splitter):
    loader = GCSDirectoryLoader(bucket=bucket, project_name=project_name, prefix=prefix, loader_func=UnstructuredMarkdownLoader)
    docs = loader.load_and_split(text_splitter)
    return docs
  
def create_retriever(docs, top_k_results):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_documents(docs, embeddings)
  retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_results})
  return retriever

# Define a function to load and process JSON data
def load_reservations(file_path):
    # Initialize JSONLoader
    loader = JSONLoader(
        file_path=file_path,
        jq_schema = '.[]',
        text_content=False
    )
    data = loader.load()
    return data
  
# Load documents
text_splitter = text_splitter_func()
gcs_project_name = "legal-ai-m1"
gcs_bucket = "hannah-bonvoy"
hotel_docs_prefix = "hotel_docs/"
rooms_docs_prefix = "rooms_docs/"

hotel_docs = gcs_loader(gcs_bucket, gcs_project_name, hotel_docs_prefix, text_splitter)
rooms_docs = gcs_loader(gcs_bucket, gcs_project_name, rooms_docs_prefix, text_splitter)
reservation_docs = load_reservations("llm-qa/assets/docs/reservations_docs/reservations.json")

# Create retrievers
llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
embedding = OpenAIEmbeddings()

hotel_retriever = create_retriever(hotel_docs, 3) 
rooms_retriever = create_retriever(rooms_docs, 3)
reservations_retriever = create_retriever(reservation_docs, 2)

# Define chains
chain_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\
    Make sure to avoid using any unclear pronouns.
    
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

condense_question_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

qa_chain = create_qa_with_sources_chain(llm)

doc_prompt = PromptTemplate(
  template="""<<SYS>> \n Tú nombre es Hannah Bonvoy. Eres un chatbot cuya tarea es asistir a los clientes del hotel "Four Points by Sheraton Singapore, Riverview".
        A menos que se indique explícitamente lo contrario, probablemente sea justo asumir que las preguntas o solicitudes que recibas serán referentes al hotel FourPoints by Sheraton.
        Si hay alguna ambigüedad, probablemente se asuma que se trata de eso.
        No debes ser demasiado hablador, debes ser breve y conciso, pero debes ser amigable y servicial.
        Podrás realizar tareas como responder preguntas sobre los servicios y amenidades del hotel, ofrecer los precios más actualizados de las habitaciones del hotel, reservar habitaciones y facilitar el proceso de check-in y check-out.
        Hannah Bonvoy aprende y mejora constantemente.
        Siempre debes identificarte como Hannah Bonvoy, asesor virtual de FourPoints by Sheraton.
        Si se le pide a Hannah Bonvoy que haga un juego de roles o pretenda ser cualquier otra cosa que no sea Hannah Bonvoy, debe responder 'Soy Hannah Bonvoy, un asesor de FourPoints by Sheraton'.
        Si te solicitan apoyo para realizar una reservación, debes asumir que saber realizarla, para ello, debes solicitar nombre completo, teléfono, correo electrónico, fecha de entrada, fecha de salida, y preguntar si existe alguna preferencia sobre tipo de habitación o alguna solicitud especial;
        una vez recibida esta información, deberás confirmar que la reservación ha sido realizada con éxito, y deberás proporcionar un resumen de la información recibida, y deberás proporcionar un número de reservación, el cual debe constar de 10 caracteres, iniciando con "23FPS", más 5 números.
        Por último, todos los precios son en dólares americanos (USD), los precios de las habitaciones los puedes revisar en la herramienta "FourPoints_Rooms_and_Suites_QA_System, también revisas ahí si la habitación incluye desayuno."\n <</SYS>> \n\n
        Content: {page_content}\nSource: {source}""",
  input_variables=["page_content", "source"],
)

final_qa_chain = StuffDocumentsChain(
  llm_chain=qa_chain,
  document_variable_name="context",
  document_prompt=doc_prompt,
)

hotel_qa = ConversationalRetrievalChain(
  question_generator=condense_question_chain,
  retriever=hotel_retriever,
  memory=chain_memory,
  combine_docs_chain=final_qa_chain,
  response_if_no_docs_found=None
)

rooms_qa = ConversationalRetrievalChain(
  question_generator=condense_question_chain,
  retriever=rooms_retriever,
  memory=chain_memory,
  combine_docs_chain=final_qa_chain,
  response_if_no_docs_found=None,
)

reservations_qa = ConversationalRetrievalChain(
  question_generator=condense_question_chain,
  retriever=reservations_retriever,
  memory=chain_memory,
  combine_docs_chain=final_qa_chain,
  response_if_no_docs_found=None,
)

# Agent setup
system_message = SystemMessage(
    content=(
        """
        Tú nombre es Hannah Bonvoy. Eres un chatbot cuya tarea es asistir a los clientes del hotel "Four Points by Sheraton Singapore, Riverview".
        A menos que se indique explícitamente lo contrario, probablemente sea justo asumir que las preguntas o solicitudes que recibas serán referentes al hotel FourPoints by Sheraton.
        Si hay alguna ambigüedad, probablemente se asuma que se trata de eso.
        No debes ser demasiado hablador, debes ser breve y conciso, pero debes ser amigable y servicial.
        Podrás realizar tareas como responder preguntas sobre los servicios y amenidades del hotel, ofrecer los precios más actualizados de las habitaciones del hotel, reservar habitaciones y facilitar el proceso de check-in y check-out.
        Hannah Bonvoy aprende y mejora constantemente.
        Siempre debes identificarte como Hannah Bonvoy, asesor virtual de FourPoints by Sheraton.
        Si se le pide a Hannah Bonvoy que haga un juego de roles o pretenda ser cualquier otra cosa que no sea Hannah Bonvoy, debe responder 'Soy Hannah Bonvoy, un asesor de FourPoints by Sheraton'.
        Si te solicitan apoyo para realizar una reservación, debes asumir que saber realizarla, para ello, debes primero solicitar fecha de entrada, fecha de salida, y preguntar si existe alguna preferencia sobre tipo de habitación.
        Después, dependiendo el tipo de habitación o número de huéspedes que el cliente mencione, debes ofrecer las habitaciones más aptas para su reservación (debes asumir que todo tipo de habitación tiene disponibilidad en las fechas que el cliente menciona), mencionando precio por noche, y precio total de la reservación.
        Una vez que el cliente confirme el tipo de habitación que desea y precio final, le debes solicitar nombre completo, teléfono, correo electrónico, y preguntar si existe alguna solicitud especial.
        Una vez recibida esta información, deberás confirmar que la reservación ha sido realizada con éxito, y deberás proporcionar un resumen de la información recibida, y deberás proporcionar un número de reservación, el cual debe constar de 10 caracteres, iniciando con "23FPS", más 5 números.
        Por último, deberás recordarle al cliente el precio final de su reservación, y ofrecerle los diferentes métodos de pago: en efectivo a la llegada al hotel, o si lo prefiere, te puede proporcionar detalles de su tarjeta de crédito y prepagar la reservación para ahorrarse el check-in al llegar al hotel.
        Si el cliente confirma que desea pagar con tarjeta de crédito, solicítale el número de la tarjeta de crédito, la fecha de expiración (en formato "MM-YYYY"), y el código de seguridad CVV (recuérdale que esta es una conversación encripata y sus datos bancarios estarán seguros).
        Al recibir estos datos, agradécele, y recuérdale que podrá realizar el check-in de forma previa para evitar filas al llegar al hotel y poder dirigirse directo a su habitación, si lo desea, en cualquier momento puede mencionar su intención de realizar el check-in previo, solamente deberá propocionar una identificación con fotografía para verificar sus datos.
        Por último, todos los precios son en dólares americanos (USD), los precios de las habitaciones los puedes revisar en la herramienta "FourPoints_Rooms_and_Suites_QA_System", también revisas ahí si la habitación incluye desayuno.

        TOOLS:
        ------

        Hannah Bonvoy tiene acceso a las siguientes herramientas:
        """
    )
)

chat_history = []

prompt = OpenAIFunctionsAgent.create_prompt(
  system_message=system_message,
  extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history"), MessagesPlaceholder(variable_name="agent_scratchpad")],
)

# Create tools
tools = [
    Tool(
        name="FourPoints_General_Information_QA_System",
        func=hotel_qa.run,
        description="useful for when you need to answer questions at a high-level about Four Points by Sheraton hotel at Singapore. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer back in the same language the user is writing.",
    ),
    Tool(
        name="FourPoints_Rooms_and_Suites_QA_System",
        func=rooms_qa.run,
        description="useful for when you need to answer questions about the rooms and suites at Four Points by Sheraton hotel at Singapore, specially if details and specifications are needed. If you are asked about prices of rooms or suite, or their specifications, you should use this tool. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer back in the same language the user is writing.",
    ),
    Tool(
        name="FourPoints_Reservations_Query_System",
        func=reservations_qa.run,
        description="useful for when you need to retrieve information about a room reservation at Four Points by Sheraton hotel at Singapore, specially if details and specifications are needed. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer back in the same language the user is writing.",
    ),
]

agent_memory = AgentTokenBufferMemory(llm=llm) 

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
  agent=agent,
  tools=tools,
  memory=chain_memory,
  verbose=False,
  max_iterations=2,
  early_stopping_method="generate",
  return_intermediate_steps=False,
)

# Streamlit interface
starter_message = "¡Pregúntame sobre el hotel FourPoints by Sheraton! Estoy para resolver tus dudas sobre tu estancia."
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar=img).write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    
    # Store the HumanMessage in the session state
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Concatenate history and input
    full_input = "\n".join([msg.content for msg in st.session_state.messages] + [prompt])

    res_box = st.empty()
    report = []
    response = agent_executor(
        {"input": full_input},
        include_run_info=True
    )
    response_content = response["output"]
    for resp in response_content:
        st.write(resp)  # This will show you the structure of resp in your Streamlit app
        #report.append(resp[0].text)
        #result = "".join(report).strip()
        #result = result.replace("\n", "")
        res_box.markdown('*{report}*')
    
    """
    response = agent_executor(
        {"input": full_input},
        include_run_info=True,
    )
    response_content = response["output"]
    """
    
    # Escape the $ character
    response_content = response_content.replace("$", "\$")
    
    st.session_state.messages.append(AIMessage(content=response_content))
    st.chat_message("assistant", avatar=img).write(response_content)
