# API import Section
import copy
from fastapi import FastAPI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langchain.utilities import SerpAPIWrapper
from langchain.tools import Tool

from utils import *

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="MoneyMaker API for GPT 3.5",
    description="A simple REST API support by a streamlit interface",
    version="0.999",
)

def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for i in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[i]
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = load_db(embedding_function=embeddings, persistence_directory='demodb/')
    return vectorstore

# get pdf text
raw_text = get_pdf_text('TEST_DOC.pdf')
# get the text chunks
text_chunks = get_text_chunks(raw_text)
 # create vector store
vectorstore = get_vectorstore(text_chunks).as_retriever(search_type = 'similarity', search_kwargs={"k": 3})

#load LLM
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
# compressor retriver
compressor = LLMChainExtractor.from_llm(llm=llm)
compressor_retriver = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=vectorstore)
#create a chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compressor_retriver,
)


search = SerpAPIWrapper()
tools = [
    Tool(
        name="Document_Store",
        func=qa.run,
        description="Use it to lookup information from the document store. Always used as first tool."
    ),
    Tool(
        name='Search',
        func=search.run,
        description='Use this to lookup information from google search engine. Use it only after you have tried using the Document_Store_tool.'
    )
]

memory = AgentTokenBufferMemory(memory_key='history', llm=llm)
message = SystemMessage(
    content=(
    """Answer the following questions as best you can. \
    You have access to the following tools:

    Document_Store: Use it to lookup information from document store. \
                    Always used as first tool
    Search: Use this to lookup information from google search engine. \
            Use it always after you have tried using the Document_Store tool.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [Document_Store, Search]. \
            Always look first in Document_Store.
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question.

    Use three sentences maximum. Keep the answer as concise as possible.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)



@app.get('/')
async def hello():
    return {"hello" : "world"}

@app.get('/model')
async def model(question : str):
    # res = chat.run(question)
    res = agent_executor({"input": question})
    result = copy.deepcopy(res)
    return result