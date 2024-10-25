import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from openai.embeddings_utils import EmbeddingsUtils
from openai.api_base import APIBase
from openai.error import OpenAIError
import openai

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = EmbeddingsUtils.create_embedding_model()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question in Bengali as accurately as possible. If the answer is not available in the context, say "answer is not available in the context". Do not provide wrong answers.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = APIBase.create_model("mpt-7b")  # or "mpt-13b"
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

    return chain



def user_input(user_question):
    
    new_db = FAISS.load_local("faiss_index", allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Study Helping AI")
    st.header("Study Helping AI")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        try:
            user_input(user_question)
        except OpenAIError as e:
            st.error(f"An error occurred: {e}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
