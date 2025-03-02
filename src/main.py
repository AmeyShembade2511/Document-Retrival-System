from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from document import Document
from vector_store import VectorStore
from mock_llm import MockLLM
from process_query import process_query
import os
import logging
import shutil
from contextlib import asynccontextmanager
import glob




# Get the current directory (assuming it's 'src')
current_dir = os.getcwd()

# Move one directory up and then into 'data'
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))



# Ensure the upload folder exists
os.makedirs(data_dir, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cleanup on shutdown"""
    yield
    files = glob.glob(os.path.join(data_dir, "*"))  # Get all files
    for file in files:
        os.remove(file)  # Delete each file
    print("Uploads folder cleared on server shutdown.")
# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Global objects for processing
vector_store = VectorStore()
mock_llm = MockLLM()
doc_processor = None  # Will store the document processor object

class UploadRequest(BaseModel):
    file_path: str  # PDF/TXT file path

class QueryRequest(BaseModel):
    query: str  # User query

@app.post("/upload")
def upload_document(request: UploadRequest):
    """Uploads and processes a document (PDF/TXT) with error handling."""
    global doc_processor

    if not os.path.exists(request.file_path):
        logging.error(f"File not found: {request.file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    if not (request.file_path.endswith(".txt") or request.file_path.endswith(".pdf")):
        logging.error(f"Invalid file format: {request.file_path}")
        raise HTTPException(status_code=400, detail="Unsupported file format. Use TXT or PDF.")

    try:
        shutil.copy(request.file_path, data_dir)
        
        doc_processor = Document(request.file_path)

        if not doc_processor.chunks:
            raise ValueError("Document has no readable content.")

        # Store embeddings in the vector store
        vector_store.add_embeddings(doc_processor.embeddings,
                                    metadata=[{"chunk": chunk} for chunk in doc_processor.chunks])

        logging.info(f"Document {request.file_path} processed successfully.")
        return {"message": f"Document {request.file_path} processed successfully", 
                "total_chunks": len(doc_processor.chunks)}

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the document.")

@app.post("/query")
def query_document(request: QueryRequest):
    """Processes a query and returns relevant document excerpts with error handling."""
    if doc_processor is None:
        logging.warning("Query received but no document uploaded.")
        raise HTTPException(status_code=400, detail="No document uploaded. Please upload a document first.")

    if not request.query.strip():
        logging.warning("Empty query received.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = process_query(request.query, vector_store, mock_llm, doc_processor.vectorizer)
        return response

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")
