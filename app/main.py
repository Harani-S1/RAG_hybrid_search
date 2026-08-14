from fastapi import FastAPI, HTTPException

from app.schemas import QuestionRequest, QuestionResponse

from src.generation.rag_pipeline import (
    create_bm25_index,
    rag_answer,
)


app = FastAPI(
    title="RAG Hybrid Search API",
    description=(
        "RAG system using Dense Retrieval, BM25, "
        "Hybrid RRF, Cross-Encoder Reranking and Groq"
    ),
    version="1.0.0",
)


# ============================================================
# GLOBAL RAG RESOURCES
# ============================================================

bm25 = None
chunks = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    global bm25, chunks

    print("\n========================================")
    print("       INITIALIZING RAG SYSTEM")
    print("========================================")

    print("\nCreating BM25 index...")

    bm25, chunks = create_bm25_index()

    print("\nBM25 index ready.")
    print(f"Chunks loaded: {len(chunks)}")

    print("\nRAG system ready.")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "RAG Hybrid Search API is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    if bm25 is None or chunks is None:
        return {
            "status": "starting",
            "rag_ready": False,
        }

    return {
        "status": "healthy",
        "rag_ready": True,
        "chunks": len(chunks),
    }


# ============================================================
# ASK
# ============================================================

@app.post(
    "/v1/ask",
    response_model=QuestionResponse,
)
def ask_question(request: QuestionRequest):

    if bm25 is None or chunks is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        print("\n========================================")
        print("           API QUESTION")
        print("========================================")

        print(f"Question: {question}")

        answer = rag_answer(
            question=question,
            bm25=bm25,
            chunks=chunks,
        )

        return QuestionResponse(
            question=question,
            answer=answer,
        )

    except Exception as error:

        print("\n========================================")
        print("              ERROR")
        print("========================================")

        print(type(error).__name__)
        print(error)

        raise HTTPException(
            status_code=500,
            detail="Failed to generate RAG answer.",
        )