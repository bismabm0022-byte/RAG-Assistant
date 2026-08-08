from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any

from document_loader import load_documents
from text_splitter import split_documents
from vector_store import build_or_load_vector_store
from retrieval import LegalRAGPipeline

app = FastAPI(title="Pakistan Law Assistant API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = LegalRAGPipeline()

class QueryRequest(BaseModel):
    question: str

@app.post("/ingest")
def ingest():
    try:
        raw_docs = load_documents()
        chunks = split_documents(raw_docs)
        build_or_load_vector_store(chunks)
        return {"status": "success", "message": f"Ingested {len(chunks)} legal chunks into vector database."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        res = pipeline.query(req.question)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
