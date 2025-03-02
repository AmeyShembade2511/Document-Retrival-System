# Document Retrieval System

This is a **Document Retrieval System** built using **FastAPI**. The system allows users to upload text or PDF documents, process them into word-based chunks, generate embeddings using **Sentence Transformers**, store them in a vector store, and retrieve relevant excerpts using **semantic similarity**.

## Features
- **Upload PDFs/TXT files** and extract text.
- **Chunk text into overlapping word-based segments** for better retrieval.
- **Generate embeddings using `SentenceTransformers` (`all-mpnet-base-v2`)**.
- **Vector storage and retrieval using cosine similarity**.
- **Rate-limited query processing to prevent excessive API calls**.
- **FastAPI-based API with `/upload` and `/query` endpoints**.

---

## Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone <repository_link>
```

### 2️⃣ Navigate to the `src` Directory
```bash
cd <repository_folder>/src
```

### 3️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 4️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```



### 5️⃣ Run the FastAPI Server
```bash
uvicorn main:app --reload
```
OR

```
python -m uvicorn main:app --reload (For Windows)
```

This will start the server at `http://127.0.0.1:8000`.

---

## API Usage Examples

### 📌 **1. Upload a Document**
**Endpoint:**
```http
POST http://127.0.0.1:8000/upload
```
**Body (JSON):**
```json
{
  "file_path": "C:\\Downloads\\BE_Survey_Paper_NTAI.pdf"
}
```
**Response:**
```json
{
  "message": "Document C:\\Downloads\\BE_Survey_Paper_NTAI.pdf processed successfully",
  "total_chunks": 15
}
```

### 📌 **2. Query the Document**
**Endpoint:**
```http
POST http://127.0.0.1:8000/query
```
**Body (JSON):**
```json
{
  "query": "What are the challenges for NER?"
}
```
**Response:**
```json
{
  "response": "Based on context, here are the most relevant pieces:\n\n1. Code-switching complexity in social media.\n2. Scarcity of annotated datasets for Hindi-English NER.\n3. Informal language characteristics.\n\nThese seem relevant to your question: What are the challenges for NER?",
  "status": "success"
}
```

---

## Design Decisions Explanation

### 📌 **1. Text Processing**
- **Text is extracted** from PDF/TXT files.
- **Chunking is word-based**, ensuring context preservation.

### 📌 **2. Sentence Embeddings**
- Used **`SentenceTransformer(all-mpnet-base-v2)`** instead of BERT for better performance in retrieval tasks.

### 📌 **3. Vector Storage & Retrieval**
- Chunks are stored as embeddings.
- **Cosine similarity** is used for **efficient retrieval**.

### 📌 **4. Rate Limiting**
- The **MockLLM** enforces **10 queries per minute** to prevent excessive API calls.

---

## Limitations and Potential Improvements

### ❌ Limitations
1. **Does not use an actual LLM** for query processing (currently uses a mock response generator).
2. **Limited to local file storage** (`data/` folder) – no cloud integration.
3. **No advanced re-ranking** – results are retrieved based purely on cosine similarity.

### ✅ Potential Improvements
1. **Integrate a real LLM (e.g., OpenAI GPT, Hugging Face models).**
2. **Use FAISS for better large-scale retrieval.**
3. **Enable multi-document support.**
4. **Enhance chunking methods for better context capture.**
5. **Deploy the system on cloud (e.g., AWS, Azure).**

---

## Folder Structure
```
📂 document_retrieval_system
│── 📂 src
│   ├── 📜 main.py            # FastAPI server
│   ├── 📜 document.py        # Document processing
│   ├── 📜 vector_store.py    # Vector storage and retrieval
│   ├── 📜 process_query.py   # Query processing
│   ├── 📜 mock_llm.py        # Mock LLM for response generation
│── 📂 data                   # Stores uploaded documents
│── 📜 requirements.txt       # Python dependencies
│── 📜 README.md              # Project documentation
```

---

## 🔥 Conclusion
This **Document Retrieval System** enables efficient semantic search over text/PDF files. It serves as a foundation for **RAG-based applications**, knowledge retrieval, and research analysis. 🚀

