import os
from dotenv import load_dotenv
#from langchain_community.llms import Ollama
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT_NAME")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


from langchain_ollama import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question below"),
        ("user", "Question:{question}")
    ]
)   

st.title("Ollama with Langchain with Gemma2 model")
input_text = st.text_input("Enter your question here:") 

llm = OllamaLLM(model="gemma3:1b")
output_parser = StrOutputParser()
chain = prompt |llm | output_parser


if input_text:
    st.write(chain.invoke({"question": input_text}))


 






