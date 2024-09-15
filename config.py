from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Ollama model configurations
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    CODE_UNDERSTANDING_MODEL: str = "codellama:7b"  # Model for final code understanding
    HYDE_MODEL: str = "codellama:7b"  # Model for generating hypothetical answers
    EMBEDDING_MODEL: str = "nomic-embed-text"  # Embedding model from Ollama
    
    # Retrieval configurations
    VECTOR_TOP_K: int = 2  # Number of chunks to retrieve from vector search
    BM25_TOP_K: int = 2    # Number of chunks to retrieve from BM25 search
    
    class Config:
        env_file = ".env"

settings = Settings()
