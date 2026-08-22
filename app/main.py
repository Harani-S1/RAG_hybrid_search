from fastapi import FastAPI, HTTPException
from pathlib import Path
import pymupdf

from app.schemas import QuestionRequest, QuestionResponse

from src.generation.rag_pipeline import (
    create_bm25_index,
    rag_answer,
)


# ============================================================
# FASTAPI APP
# ============================================================

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
# DOCUMENTS
# ============================================================

@app.get("/v1/documents")
def get_documents():

    data_dir = Path("data/raw")

    if not data_dir.exists():
        return {
            "documents": [],
            "count": 0,
        }

    documents = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):

        try:
            with pymupdf.open(pdf_path) as pdf:
                page_count = len(pdf)

            documents.append({
                "name": pdf_path.name,
                "path": str(pdf_path).replace("\\", "/"),
                "pages": page_count,
            })

        except Exception as error:

            print(
                f"Failed to read {pdf_path}: {error}"
            )

    return {
        "documents": documents,
        "count": len(documents),
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

        # ====================================================
        # RUN COMPLETE RAG PIPELINE
        # ====================================================

        result = rag_answer(
            question=question,
            bm25=bm25,
            chunks=chunks,
            return_details=True,
        )

        # ====================================================
        # GET ANSWER
        # ====================================================

        answer = result.get(
            "answer",
            "No answer returned.",
        )

        # ====================================================
        # GET RETRIEVED RESULTS
        # ====================================================

        retrieved_results = result.get(
            "retrieved_results",
            [],
        )

        retrieved_chunk_ids = result.get(
            "retrieved_chunk_ids",
            [],
        )

        # ====================================================
        # BUILD CITATIONS
        # ====================================================

        citations = []

        for item in retrieved_results:

            metadata = item.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            source = metadata.get(
                "source",
                "Unknown",
            )

            page = metadata.get(
                "page",
                None,
            )

            human_page = metadata.get(
                "human_page",
                None,
            )

            reranker_score = item.get(
                "reranker_score",
                None,
            )

            citation = {
                "source": str(source),
                "page": page,
                "human_page": human_page,
            }

            if reranker_score is not None:

                citation["reranker_score"] = float(
                    reranker_score
                )

            citations.append(citation)

        # ====================================================
        # UNIQUE SOURCES
        # ====================================================

        unique_sources = []

        for citation in citations:

            source = citation.get("source")

            if (
                source
                and source not in unique_sources
            ):

                unique_sources.append(
                    source
                )

        # ====================================================
        # CALCULATE CONFIDENCE
        # ====================================================

        confidence = None

        scores = []

        for item in retrieved_results:

            score = item.get(
                "reranker_score"
            )

            if score is not None:

                scores.append(
                    float(score)
                )

        if scores:

            average_score = (
                sum(scores) / len(scores)
            )

            # Convert reranker score to a
            # dashboard confidence value.

            import math

            confidence = (
                1
                / (
                    1
                    + math.exp(
                        -average_score
                    )
                )
            ) * 100

            confidence = max(
                0,
                min(
                    100,
                    confidence,
                ),
            )

        # ====================================================
        # PRINT RESPONSE INFORMATION
        # ====================================================

        print("\n========================================")
        print("           RAG RESPONSE")
        print("========================================")

        print(
            f"Answer generated: {bool(answer)}"
        )

        print(
            f"Citations: {len(citations)}"
        )

        print(
            f"Sources: {len(unique_sources)}"
        )

        print(
            f"Confidence: {confidence}"
        )

        print(
            f"Retrieved chunks: "
            f"{len(retrieved_chunk_ids)}"
        )

        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return QuestionResponse(
            question=question,
            answer=answer,
            citations=citations,
            sources=unique_sources,
            confidence=confidence,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        print("\n========================================")
        print("              ERROR")
        print("========================================")

        print(
            type(error).__name__
        )

        print(error)

        raise HTTPException(
            status_code=500,
            detail="Failed to generate RAG answer.",
        )