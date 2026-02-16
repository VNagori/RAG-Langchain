
import streamlit as st
from langchain_ollama import OllamaLLM

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload PDFs and chat with them!")

api_key = st.text_input("Enter your Groq API Key", type="password")
if api_key:
    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
    session_id = st.text_input("Session ID for chat history", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}
        
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True) 
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                f_name = uploaded_file.name
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        split_docs = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(split_docs, embeddings)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        contextualize_q_system_prompt = (
            "Given a chat history and a new question, rephrase the new question to be a standalone question"
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])


        #history_runnable = RunnableChatMessageHistory(st.session_state.chat_history)

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        system_prompt = (
            "You are a helpful AI assistant that provides answers based on the provided context\n\n{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt_template
        )

        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=document_chain
        )

        def get_session_chat_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversation_rag_history = RunnableWithMessageHistory(  
            rag_chain,
            get_session_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_question = st.text_input("Enter your question about the PDFs:")

        if user_question:
            session_history = get_session_chat_history(session_id)

            response = conversation_rag_history.invoke(
                    {"input" : user_question},
                    config = {
                        "configurable" : { "session_id" : session_id}
                    },
            )


            st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")
            st.write("Chat History:", session_history.messages)
            # for doc in response['source_documents']:
            #     st.write(doc.page_content)

            # st.session_state.chat_history.add_user_message(user_question)
            # st.session_state.chat_history.add_ai_message(response['output'])
    else:
        st.write("Please upload at least one PDF file to proceed.")
else:
    st.write("Please enter your Groq API Key to proceed.")


