import numpy as np
import logging
from vector_store import VectorStore
from mock_llm import MockLLM
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_query(query: str, vector_store: VectorStore, llm: MockLLM, vectorizer: SentenceTransformer) -> dict:
    """Processes query, retrieves relevant chunks, and handles errors gracefully."""
    
    try:
        # Handle empty query
        if not query.strip():
            logging.warning("Received an empty query.")
            return {"error": "Query cannot be empty.", "status": "failed"}

        # Ensure embeddings exist
        if not vector_store.embeddings:
            logging.warning("No documents available in the vector store.")
            return {"error": "No documents available in the vector store.", "status": "failed"}

        # Generate query embedding
        try:
            query_embedding = vectorizer.encode([query], convert_to_numpy=True)[0]
        except Exception as e:
            logging.error(f"Error generating query embedding: {e}")
            return {"error": "Failed to generate query embedding.", "status": "failed"}

        # Convert query embedding to tuple (hashable)
        query_embedding_tuple = tuple(query_embedding.tolist())

        # Perform search
        results = vector_store.search(query_embedding_tuple)

        if not results:
            logging.warning(f"No relevant documents found for query: {query}")
            return {"error": "No relevant documents found.", "status": "failed"}

        # Convert context to tuple
        context = tuple(r["chunk"] for r in results)

        # Generate response using Mock LLM
        response = llm.generate_response(query, context)
        return response

    except Exception as e:
        logging.error(f"Unexpected error processing query: {e}")
        return {"error": "An internal error occurred while processing the query.", "status": "failed"}
