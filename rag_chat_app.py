import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain import hub

import os
import bs4
import tempfile

# Use environment variables without setting them in the code
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Optionally, you can add checks to ensure the environment variables are set
if not all([LANGCHAIN_API_KEY, OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables. Please set LANGCHAIN_API_KEY and OPENAI_API_KEY.")

# Streamlit app: Create the user interface
st.title("RAG with Streamlit")

# File upload option
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])

# Input field for URL
url_input = st.text_input("Or enter a URL:")

if uploaded_file is not None or url_input:
    # Load Documents based on input type
    if uploaded_file is not None:
        # Handle uploaded file
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == 'txt':
            loader = TextLoader(uploaded_file.getvalue().decode())
        elif file_extension == 'pdf':
            from langchain_community.document_loaders import PyPDFLoader
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == 'docx':
            from langchain_community.document_loaders import Docx2txtLoader
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            loader = Docx2txtLoader(temp_file_path)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            st.stop()
    else:
        # Handle URL input
        loader = WebBaseLoader(
            web_paths=(url_input,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "span"]
                )
            ),
        )

    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Check if splits is empty
    if not splits:
        st.error("No content was extracted from the input. Please try a different file or URL.")
        st.stop()

    # Embed
    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Input field for user questions
    question = st.text_input("Enter your question:")

    # Button to trigger the answer generation
    if st.button("Get Answer"):
        if question:
            # If a question is provided, invoke the RAG chain and display the result
            result = rag_chain.invoke(question)
            st.write("Answer:", result)
        else:
            # If no question is provided, prompt the user to enter one
            st.write("Please enter a question.")
else:
    st.write("Please upload a file or enter a URL to proceed.")