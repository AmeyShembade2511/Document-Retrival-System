import os
import PyPDF2
import numpy as np
import nltk
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# Load a pre-trained model
nltk.download('punkt')

class Document:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content = self.load_document()
        self.chunks = self.chunk_text()  # Use sentence-wise chunking
        self.vectorizer = SentenceTransformer('all-mpnet-base-v2')
        self.embeddings = self.generate_embeddings(self.chunks)

    def load_document(self) -> str:
        """Loads document content from PDF or TXT and removes unnecessary newlines."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Error: File '{self.file_path}' not found.")

        try:
            if self.file_path.endswith('.txt'):
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                    if not text:
                        raise ValueError("Error: The TXT file is empty.")
                    return " ".join(text.splitlines())  # ✅ Replaces \n with space

            elif self.file_path.endswith('.pdf'):
                with open(self.file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)

                    if len(reader.pages) == 0:
                        raise ValueError("Error: The PDF file has no readable pages.")

                    text = " ".join(
                        page.extract_text().replace("\n", " ") for page in reader.pages if page.extract_text()
                    )

                    if not text.strip():
                        raise ValueError("Error: No readable text found in the PDF file.")

                    return text  # ✅ Removes newlines from PDF text

            else:
                raise ValueError("Unsupported file format. Use TXT or PDF.")

        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            return ""
        except ValueError as e:
            print(f"ValueError: {e}")
            return ""
        except Exception as e:
            print(f"Unexpected error while reading the file: {e}")
            return ""

    

    def chunk_text(self, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Splits content into overlapping word-based chunks."""
        if not self.content:
            raise ValueError("Error: No content available for chunking.")

        try:
            words = word_tokenize(self.content)

            if not words:
                raise ValueError("Error: Tokenization failed, no words found.")

            chunks = []
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk = " ".join(words[start:end])  # Join words to form chunk
                chunks.append(chunk)
                start += max(1, chunk_size - overlap)  # Move forward with overlap

            return chunks

        except Exception as e:
            print(f"Unexpected error in chunking: {e}")
            return []


    def generate_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        """Generates embeddings using SentenceTransformer."""
        if not chunks:
            raise ValueError("Error: No chunks available for embedding generation.")

        try:
            embeddings = self.vectorizer.encode(chunks, convert_to_numpy=True)  # ✅ Use encode() instead of fit/transform
            #print(chunks)
            #print(embeddings)
            return embeddings
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            return []
