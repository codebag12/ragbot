import streamlit as st  # For creating web apps
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # For OpenAI language models and embeddings
from langchain.prompts import ChatPromptTemplate  # For creating chat prompts
from langchain_core.output_parsers import StrOutputParser  # For parsing output as strings
from langchain_core.runnables import RunnablePassthrough  # For passing data through the chain
from langchain_community.vectorstores import Chroma  # For vector storage
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_community.document_loaders import WebBaseLoader, TextLoader  # For loading documents from web and text files
from langchain import hub  # For accessing pre-built prompts

import os  # For interacting with the operating system
import bs4  # For parsing HTML content
import tempfile  # For creating temporary files

# Get environment variables for API keys and settings
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create the Streamlit app interface
st.title("RAG with Streamlit")

# Add a file upload option
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])

# Add an input field for URL
url_input = st.text_input("Or enter a URL:")

# Check if a file is uploaded or URL is provided
if uploaded_file is not None or url_input:
    # Load documents based on input type (file or URL)
    if uploaded_file is not None:
        # Handle uploaded file (code for different file types is omitted for brevity)
        pass
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

    # Load the documents
    docs = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Check if any content was extracted
    if not splits:
        st.error("No content was extracted from the input. Please try a different file or URL.")
        st.stop()

    # Create embeddings and store them in a vector database
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings()  
    )

    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Get a pre-built prompt for RAG (Retrieval-Augmented Generation)
    prompt = hub.pull("rlm/rag-prompt")

    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define a function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Create an input field for user questions
    question = st.text_input("Enter your question:")

    # Add a button to trigger the answer generation
    if st.button("Get Answer"):
        if question:
            # If a question is provided, run the RAG chain and display the result
            result = rag_chain.invoke(question)
            st.write("Answer:", result)
        else:
            # If no question is provided, prompt the user to enter one
            st.write("Please enter a question.")
else:
    # Display a message if no file or URL is provided
    st.write("Please upload a file or enter a URL to proceed.")