# RAG Q&A Conversation With PDF Including Chat History
# ------------------------------------------------------
# This Streamlit app allows users to upload PDF files and interactively ask questions about their content.
# It uses LangChain's Retrieval-Augmented Generation (RAG) with chat history for context-aware Q&A.

import streamlit as st
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up environment variables for LangChain and HuggingFace
os.environ['LANGSMITH_API_KEY'] = os.getenv(
    'LANGSMITH_API_KEY')  # API key for LangChain
os.environ['LANGSMITH_TRACING'] = "true"  # Enable LangChain tracing
os.environ['LANGSMITH_PROJECT'] = os.getenv(
    'LANGSMITH_PROJECT')  # Project name for tracking
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")  # HuggingFace token

# Initialize HuggingFace embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------------
# Streamlit UI Setup
# ----------------------
st.title("Conversational RAG With PDF uplaods and chat history")
st.write("Upload Pdf's and chat with their content")

# Prompt user for Groq API key (required for LLM access)
api_key = st.text_input("Enter your Groq API key:", type="password")

# Only proceed if API key is provided
if api_key:
    # Initialize the Groq LLM with the provided API key
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Prompt for a session ID to manage chat history per user/session
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize chat history store in Streamlit session state if not present
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Allow user to upload one or more PDF files
    uploaded_files = st.file_uploader(
        "Choose A PDf file", type="pdf", accept_multiple_files=True)

    # Process uploaded PDF files
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            # Save uploaded PDF to a temporary file
            file_name = uploaded_file.name
            with open(file_name, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # Load PDF and extract its content as documents
            loader = PyPDFLoader(file_name)
            docs = loader.load()
            documents.extend(docs)

        # Split documents into smaller chunks for embedding and retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Create a vector store (Chroma) from the document chunks
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # ----------------------
        # Prompt Templates
        # ----------------------
        # System prompt for contextualizing user questions using chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create a retriever that is aware of chat history
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt)

        # System prompt for answering questions using retrieved context
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create a chain for answering questions using the LLM and prompt
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        # Combine retriever and QA chain into a RAG chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain)

        # Function to get or create chat history for a session
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Wrap the RAG chain with message history management
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User input for questions
        user_input = st.text_input("Your question:")
        if user_input:
            # Retrieve session chat history
            session_history = get_session_history(session_id)
            # Invoke the conversational RAG chain with user input and session context
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },  # constructs a key "abc123" in `store`.
            )
            # Display chat store, assistant's answer, and chat history
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    # Warn user if API key is missing
    st.warning("Please enter the GRoq API Key")
