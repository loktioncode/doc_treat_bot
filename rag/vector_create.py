from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

DATA_PATH = "books"

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
            # print(text)
    return  text


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=800)
    chunks = text_splitter.split_text(text)
    print(f"Split {len(text)} documents into {len(chunks)} chunks.")
    return chunks


def generate_data_store():
    if os.path.exists(DATA_PATH):
        # List all PDF files in the specified directory
        pdf_docs = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = split_text(raw_text)
        get_vector_store(text_chunks)
        print(f"Saved {len(text_chunks)} to chunks db.")

def main():
    generate_data_store()

if __name__ == "__main__":
    main()

