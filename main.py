#################################################################################

################
# ARCHITECTURE #
################

# File Loader - GenericLoader
# Structured Literal Output to detect language
# Chunking - RecursiveCharacterTextSplitter (from Language)
# Database - Chroma
# Embedding - jinaai/jina-embeddings-v2-base-code
# Custom Hybrid Search: On HyDE Generation
#   Vector Retriever - Default
#   Keyword Retriever - BM25
# LLM - Code Llama 7B
# Agent - Regenerate HyDE Doc till satisfied with retrieved docs

#################################################################################


from config import *
from utils.create_db import create_db
from utils.create_embeddings import create_embeddings
from utils.create_retriever import create_retriever
from utils.create_generator import create_generator


query = """
Give a brief description of this code. List out some functions used in the code.
"""


embeddings = create_embeddings()
vectorstore, chunks = create_db(embeddings)
hybrid_search_retriever = create_retriever(vectorstore, chunks)
output = create_generator(query, hybrid_search_retriever)

print(output)









