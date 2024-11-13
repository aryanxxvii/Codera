from config import *

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from typing import List

def evaluate_retrieval(relevant_documents: List, min_docs: int = 3) -> bool:
    """
    Evaluates if the retrieved documents meet the satisfaction criteria.
    """
    return len(relevant_documents) >= min_docs

def rewrite_hyde_doc(query: str, attempt: int) -> str:
    """
    Generates a more refined version of the HyDE document.
    """
    if attempt == 1:
        refinement = "Please expand the explanation, adding more technical details and examples."
    else:
        refinement = f"Attempt {attempt}: Make the explanation more focused on the core functionality and add detailed code snippets."

    return f"{query} - {refinement}"

def generate_hyde_doc(query: str, model: OllamaLLM, attempt: int = 1) -> str:
    """
    Generates the hypothetical document (HyDE doc) using the model.
    Supports rewriting based on the attempt number.
    """
    hyde_query = rewrite_hyde_doc(query, attempt)

    template = """Imagine you are an expert writing a detailed explanation on the topic: '{query}'
    Your response should be comprehensive and include all key points that would be found in the top search result."""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | model

    hyde_doc = chain.invoke({"query": hyde_query})

    return hyde_doc

def create_generator(query, retriever):
    model = OllamaLLM(model=MODEL_NAME)

    attempt = 1
    while attempt <= MAX_REGENERATION_ATTEMPTS:
        hyde_doc = generate_hyde_doc(query, model, attempt)

        relevant_documents = retriever.get_relevant_documents(query=hyde_doc)

        if evaluate_retrieval(relevant_documents):
            break

        attempt += 1

    str_relevant_documents = [doc for doc in relevant_documents]

    template = """You are provided with the following relevant documents: {str_relevant_documents}
    Answer the question: {query}"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | model

    answer = chain.invoke({"query": query, "str_relevant_documents": str_relevant_documents})

    return answer
