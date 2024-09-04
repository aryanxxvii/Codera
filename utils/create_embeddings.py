from config import *

import os
from langchain.embeddings import HuggingFaceEmbeddings


def create_embeddings():
    HF_API_TOKEN = os.environ.get("HF")

    embeddings = HuggingFaceEmbeddings(
        api_key=HF_API_TOKEN, model_name=EMBEDDER_MODEL_NAME, model_kwargs=EMBEDDER_KWARGS
    )

    return embeddings