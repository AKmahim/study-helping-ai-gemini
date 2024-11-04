import os
from dotenv import load_dotenv
from mistralai import Mistral
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter



# api_key = os.environ["MISTRAL_API_KEY"]
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

# print(api_key)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "tumi ki banglay ktha bolte paro?",
        },
    ]
)
print(chat_response.choices[0].message.content)