from config import *

import os
import shutil

from utils.detect_lang import detect_language

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def create_db(embeddings):
    document = load_document()
    chunks = chunk_text(document)
    db = save_to_db(chunks, embeddings)
    return db, chunks
    
def load_document():
    loader = TextLoader(FILE_PATH)
    document = loader.load()
    return document

def chunk_text(document):
    text_splitter = RecursiveCharacterTextSplitter(
        language=detect_language(document),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    chunks = text_splitter.split_documents(document)
    return chunks

def save_to_db(chunks, embeddings):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()

    return db