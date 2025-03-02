import time
import random
import functools
import logging
from typing import List, Dict, Tuple
from transformers import pipeline
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv







# Configure logging
logging.basicConfig(level=logging.INFO)

class MockLLM:
    def __init__(self, rate_limit: int = 10):
        self.request_count = 0
        self.rate_limit = rate_limit
        self.last_request_time = time.time()

        

   
    @functools.lru_cache(maxsize=50)  # Cache up to 50 unique queries
    def generate_response(self, prompt: str, context: Tuple[str, ...]) -> Dict:
        """Simulates LLM response with rate limiting, using top 2 relevant results. and do rate limiting."""

        try:
            # Check for empty prompt
            if not prompt.strip():
                logging.warning("Received an empty prompt.")
                return {"error": "Query cannot be empty.", "status": "failed"}

            # Check for empty context
            if not context:
                logging.warning(f"No context found for query: {prompt}")
                return {"error": "No relevant context found.", "status": "failed"}

            # Rate limiting enforcement
            current_time = time.time()
            elapsed_time = current_time - self.last_request_time  # Time since last reset

            if self.request_count >= self.rate_limit:
                if elapsed_time < 60:  # If 60 seconds have not passed
                    remaining_time = 60 - elapsed_time  # Time left before reset
                    logging.warning(f"Rate limit exceeded. Try again in {remaining_time:.2f} seconds.")
                    return {"error": f"Rate limit exceeded. Please wait {remaining_time:.2f} seconds.", "status": "failed"}
                else:
                    # Reset counter after 60 seconds
                    self.request_count = 0
                    self.last_request_time = time.time()  # Update last reset time

            self.request_count += 1  # Increase request count

            # Prepare prompt with retrieved chunks
            formatted_context = "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(context))
            final_prompt = f"Based on context, here are the most relevant pieces:\n\n{formatted_context}\n\nThese seem relevant to your question:{prompt}"
            

            return {"response": final_prompt, "status": "success"}

        except Exception as e:
            logging.error(f"Unexpected error in LLM response generation: {e}")
            return {"error": "An internal error occurred while generating the response.", "status": "failed"}
