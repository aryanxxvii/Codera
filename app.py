from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from typing import List, Dict
from config import settings
import json
import requests

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

if not check_ollama_connection():
    raise RuntimeError(
        "Ollama service is not running. Please start Ollama first using 'ollama serve' command."
    )

# Initialize embeddings
embeddings = OllamaEmbeddings(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.EMBEDDING_MODEL
)

# Initialize Ollama clients
code_llm = Ollama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.CODE_UNDERSTANDING_MODEL
)
hyde_llm = Ollama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.HYDE_MODEL
)

# Initialize retrievers
bm25_retriever = None
vector_store = None

# Language to splitter mapping
LANGUAGE_SPLITTERS = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".go": Language.GO,
    ".rs": Language.RUST,
}

def get_language_splitter(file_extension: str):
    """Get the appropriate splitter for a given file extension"""
    language = LANGUAGE_SPLITTERS.get(file_extension.lower())
    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1000,
            chunk_overlap=200
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload and process a code file"""
    try:
        # Read file content
        content = await file.read()
        content = content.decode('utf-8')
        
        # Get file extension
        file_extension = '.' + file.filename.split('.')[-1]
        
        # Get appropriate splitter
        splitter = get_language_splitter(file_extension)
        
        # Split content into chunks
        chunks = splitter.split_text(content)
        
        # Create documents
        documents = [Document(page_content=chunk, metadata={"source": file.filename}) for chunk in chunks]
        
        # Initialize BM25 retriever
        global bm25_retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = settings.BM25_TOP_K
        
        # Initialize vector store
        global vector_store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="code_chunks"
        )
        
        return {"message": f"Successfully processed {file.filename}", "num_chunks": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_hyde_answer(query: str) -> str:
    """Generate a hypothetical answer using HyDE"""
    if not check_ollama_connection():
        raise HTTPException(
            status_code=503,
            detail="Ollama service is not running. Please start Ollama first using 'ollama serve' command."
        )
        
    hyde_prompt = PromptTemplate(
        template="""Given the following question about some code, write a hypothetical answer that might be correct:
        
Question: {query}

Write a short, specific, and technical response:""",
        input_variables=["query"]
    )
    
    return hyde_llm.invoke(hyde_prompt.format(query=query))

@app.post("/query")
async def query_code(query: str):
    """Query the code database using hybrid search (BM25 + HyDE)"""
    try:
        if not bm25_retriever or not vector_store:
            raise HTTPException(status_code=400, detail="No code has been uploaded yet")
            
        if not check_ollama_connection():
            raise HTTPException(
                status_code=503,
                detail="Ollama service is not running. Please start Ollama first using 'ollama serve' command."
            )
            
        # BM25 search
        bm25_results = await bm25_retriever.ainvoke(query)
        bm25_texts = [doc.page_content for doc in bm25_results]
        
        # HyDE search
        hyde_answer = get_hyde_answer(query)
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": settings.VECTOR_TOP_K}
        )
        vector_results = await vector_retriever.ainvoke(hyde_answer)
        vector_texts = [doc.page_content for doc in vector_results]
        
        # Deduplicate chunks while preserving order
        seen = set()
        all_chunks = []
        for chunk in bm25_texts + vector_texts:
            if chunk not in seen:
                seen.add(chunk)
                all_chunks.append(chunk)
        
        # Generate final answer
        final_prompt = PromptTemplate(
            template="""You are a helpful code assistant. Based on the following code chunks, answer the user's question.
            If the code chunks don't contain enough information to answer the question fully, mention that in your response.

Code chunks:
{chunks}

Question: {query}

Please provide a clear and technical explanation:""",
            input_variables=["chunks", "query"]
        )
        
        final_answer = code_llm.invoke(
            final_prompt.format(
                chunks="\n\n".join(all_chunks),
                query=query
            )
        )
        
        return {
            "answer": final_answer,
            "hyde_answer": hyde_answer,
            "bm25_chunks": bm25_texts,
            "vector_chunks": vector_texts,
            "deduplicated_chunks": all_chunks,
            "total_unique_chunks": len(all_chunks)
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chunks")
async def get_chunks():
    """Get all code chunks currently stored in memory"""
    try:
        if not vector_store:
            raise HTTPException(status_code=400, detail="No code has been uploaded yet")
        
        # Get all documents from the vector store using as_retriever
        retriever = vector_store.as_retriever()
        docs = await retriever.ainvoke("")  # Empty query to get all docs
        
        # Format the response
        chunks = []
        for doc in docs:
            chunks.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "total_chunks": len(chunks),
            "chunks": chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
