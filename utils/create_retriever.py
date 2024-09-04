from config import *

from langchain.retrievers import BM25Retriever, EnsembleRetriever

def create_retriever(vectorstore, chunks):

    vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver, keyword_retriever],
                                           weights=[0.6, 0.4])

    return ensemble_retriever
