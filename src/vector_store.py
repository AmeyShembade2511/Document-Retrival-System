import numpy as np
import logging
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import functools

# Configure logging
logging.basicConfig(level=logging.INFO)

class VectorStore:
    def __init__(self):
        self.embeddings = []
        self.metadata = []

    def add_embeddings(self, embeddings: List[np.ndarray], metadata: List[Dict]):
        """Stores embeddings with their metadata and handles errors gracefully."""
        try:
            # Explicitly check for empty lists
            if len(embeddings) == 0 or len(metadata) == 0:
                raise ValueError("Error: Embeddings and metadata cannot be empty.")

            if len(embeddings) != len(metadata):
                raise ValueError("Error: The number of embeddings must match the number of metadata entries.")

            self.embeddings.extend(embeddings)
            self.metadata.extend(metadata)

            logging.info(f"Added {len(embeddings)} embeddings to the vector store.")

        except Exception as e:
            logging.error(f"Failed to add embeddings: {e}")


    def _hashable_query(self, query_embedding: np.ndarray) -> Tuple:
        """Converts numpy array to a hashable tuple."""
        try:
            if not isinstance(query_embedding, np.ndarray):
                raise TypeError("Error: query_embedding must be a numpy array.")
            return tuple(query_embedding.tolist())  # Convert array to tuple
        except Exception as e:
            logging.error(f"Error converting query embedding: {e}")
            return tuple()

    @functools.lru_cache(maxsize=50)  # Cache up to 50 queries
    def search(self, query_embedding_tuple: Tuple, k: int = 5, threshold: float = 0.2) -> List[Dict]:
        """Finds top-k most similar chunks using cosine similarity (cached) with error handling."""
        try:
            # Convert back to numpy array
            query_embedding = np.array(query_embedding_tuple)

            if not self.embeddings:
                logging.warning("No embeddings found in the vector store.")
                return [{"chunk": "No relevant documents found.", "score": 0.0}]

            if query_embedding.ndim != 1:
                raise ValueError("Error: Query embedding must be a 1D numpy array.")

            # Compute cosine similarity
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]

            # Pair scores with metadata
            results = [
                {"chunk": self.metadata[i]["chunk"], "score": similarities[i]}
                for i in range(len(similarities))
                if similarities[i] > threshold  # Filter low scores
            ]

            # Sort by similarity (highest first)
            results = sorted(results, key=lambda x: x["score"], reverse=True)

            # Return top-k results
            return results[:k] if results else [{"chunk": "No relevant documents found.", "score": 0.0}]

        except ValueError as ve:
            logging.error(f"ValueError in search: {ve}")
            return [{"chunk": "Invalid query embedding.", "score": 0.0}]

        except Exception as e:
            logging.error(f"Unexpected error in search: {e}")
            return [{"chunk": "An internal error occurred while searching.", "score": 0.0}]
