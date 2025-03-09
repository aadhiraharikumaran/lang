import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import os
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@st.cache_resource
def initialize_groq_llm():
    # Initialize the Groq language model with API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("API key is missing. Please check your .env file.")
        return None
    return Groq(api_key=api_key)

@st.cache_resource
def create_vector_store(pdf_path, _embeddings, store_name):
    # Extract text from PDF and create vector store
    pdf_reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text)
    vector_store = FAISS.from_texts(chunks, embedding=_embeddings)
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store

def main():
    # Main function to run the Streamlit app
    st.title("Simple RAG Application")

    llm = initialize_groq_llm()
    if llm is None:
        return  # Exit if the API key is not available

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        pdf_path = uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Define embedding models
        embeddings_models = {
            '300-dim': HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            '700-dim': HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"),
            '1536-dim': HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        }

        # Load vector stores for each embedding model
        vector_stores = {}
        for name, embeddings in embeddings_models.items():
            vector_stores[name] = create_vector_store(pdf_path, embeddings, f"vector_store_{name}")

        query = st.text_input("**Ask your question:**")
        selected_embedding = st.selectbox("Select embedding model", list(embeddings_models.keys()))

        if query and selected_embedding:
            vector_store = vector_stores[selected_embedding]
            docs = vector_store.similarity_search(query=query, k=3)

            if not docs:
                response_content = "No documents found."
            else:
                snippets = " ".join([doc.page_content for doc in docs])

                # Generate a response from the Groq language model
                prompt = f"Given the following document snippets, provide a detailed and relevant response to the query: '{query}'.\n\nDocument Snippets:\n{snippets}"
                try:
                    result = llm.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "Provide detailed and clear responses based on the provided document snippets."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    response_content = result.choices[0].message.content
                except Exception as e:
                    response_content = f"An error occurred while generating the response: {e}"

            # Display response
            st.subheader("Answer:")
            st.write(response_content)

if __name__ == '__main__':
    main()
