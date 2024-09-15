# Codera: Hybrid RAG for Code QnA

## Project Overview
Codera is a Hybrid Retrieval Augmented Generation (RAG) system built to help query code files efficiently. It uses LangChain for retrieval logic and Chroma as the vector-store to store document embeddings. Codera combines different search techniques to retrieve relevant code snippets based on queries.

## Architecture
- **LangChain:** Handles the retrieval logic.
- **Chroma Vector-Store:** Stores document embeddings for vector-based search.
- **Hybrid Search System:**
  - **HyDE Generation:** Helps generate more context for better search results.
  - **Vector Retriever:** Fetches documents based on vector similarity.
  - **BM25 Keyword Retriever:** Performs keyword-based search to supplement the vector search.
- **Code Llama 7B:** A large language model from HuggingFace, used to process and respond to code-related queries.

The system also uses structured output for language detection to make sure the right approach is used based on the code’s language.

## Why This Architecture?
- **Hybrid Retrieval:** Using both vector-based and keyword-based retrieval improves the relevance of search results.
- **Efficient Querying:** Chroma helps retrieve documents quickly, even with larger codebases.
- **Code-Specific LLM:** Code Llama 7B is optimized for working with code, making it a good fit for handling code-related queries.

## Use Case
Codera is useful for querying large codebases, understanding code, or finding specific functions and methods without manually searching through files.
