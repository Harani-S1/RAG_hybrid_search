
# ============================================================
# RAG PIPELINE
# Dense Retrieval + BM25 + RRF + CrossEncoder + Groq
#
# Improved retrieval:
# - Larger dense/BM25 candidate pools
# - RRF hybrid retrieval
# - Cross-encoder reranking
# - Low-relevance filtering
# - Stricter answer generation
# - Avoids sending obviously irrelevant chunks to Groq
# ============================================================

from sentence_transformers import CrossEncoder

from src.retrieval.dense_retriever import dense_search

from src.retrieval.bm25_retriever import (
    bm25_search,
    create_bm25_index,
)

from src.generation.groq_llm import (
    generate_answer,
)


# ============================================================
# CONFIGURATION
# ============================================================

DENSE_TOP_K = 20
BM25_TOP_K = 20
HYBRID_TOP_K = 20

# Number of chunks finally given to the LLM.
RERANK_TOP_K = 5

# Minimum CrossEncoder relevance score.
# Chunks below this are considered weak matches.
MIN_RERANK_SCORE = -1.5

RRF_K = 60

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# GLOBAL RERANKER
# ============================================================

_reranker = None


def get_reranker():

    global _reranker

    if _reranker is None:

        print(
            "\nLoading reranker model..."
        )

        _reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print(
            "Reranker model loaded successfully."
        )

    return _reranker


# ============================================================
# RESULT TEXT
# ============================================================

def get_result_text(result):

    if not isinstance(result, dict):
        return ""

    text = result.get("text")

    if text:
        return str(text)

    document = result.get("document")

    if document is not None:

        if hasattr(
            document,
            "page_content",
        ):
            return str(
                document.page_content
            )

        if isinstance(
            document,
            str,
        ):
            return document

    page_content = result.get(
        "page_content"
    )

    if page_content:
        return str(page_content)

    return ""


# ============================================================
# RESULT KEY
# ============================================================

def get_result_key(result):

    if not isinstance(result, dict):
        return ""

    result_id = result.get("id")

    if result_id:
        return str(result_id)

    metadata = result.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    source = metadata.get(
        "source",
        "",
    )

    page = metadata.get(
        "page",
        "",
    )

    text = get_result_text(result)

    return (
        f"{source}|"
        f"{page}|"
        f"{text[:200]}"
    )


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k=HYBRID_TOP_K,
    k=RRF_K,
):

    fused = {}

    # --------------------------------------------------------
    # Dense results
    # --------------------------------------------------------

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):

        key = get_result_key(result)

        if not key:
            continue

        if key not in fused:

            fused[key] = {
                "id": result.get("id"),
                "text": get_result_text(result),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "document": result.get(
                    "document"
                ),
                "dense_rank": rank,
                "bm25_rank": None,
                "dense_score": result.get(
                    "score"
                ),
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        fused[key]["rrf_score"] += (
            1.0 / (k + rank)
        )

    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        key = get_result_key(result)

        if not key:
            continue

        if key not in fused:

            fused[key] = {
                "id": result.get("id"),
                "text": get_result_text(result),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "document": result.get(
                    "document"
                ),
                "dense_rank": None,
                "bm25_rank": rank,
                "dense_score": None,
                "bm25_score": result.get(
                    "score"
                ),
                "rrf_score": 0.0,
            }

        else:

            fused[key]["bm25_rank"] = rank

            fused[key]["bm25_score"] = (
                result.get("score")
            )

            if not fused[key].get("text"):

                fused[key]["text"] = (
                    get_result_text(result)
                )

            if not fused[key].get(
                "document"
            ):

                fused[key]["document"] = (
                    result.get("document")
                )

            if not fused[key].get(
                "metadata"
            ):

                fused[key]["metadata"] = (
                    result.get(
                        "metadata",
                        {},
                    )
                )

        fused[key]["rrf_score"] += (
            1.0 / (k + rank)
        )

    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    results = sorted(
        fused.values(),
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def create_hybrid_results(
    query,
    bm25,
    chunks,
    top_k=HYBRID_TOP_K,
):

    print(
        "\n=============================="
    )

    print(
        "RUNNING DENSE RETRIEVAL"
    )

    print(
        "=============================="
    )

    dense_results = dense_search(
        query,
        top_k=DENSE_TOP_K,
    )

    if dense_results is None:
        dense_results = []

    print(
        f"Dense results: "
        f"{len(dense_results)}"
    )

    print(
        "\n=============================="
    )

    print(
        "RUNNING BM25 RETRIEVAL"
    )

    print(
        "=============================="
    )

    bm25_results = bm25_search(
        bm25,
        chunks,
        query,
        top_k=BM25_TOP_K,
    )

    if bm25_results is None:
        bm25_results = []

    print(
        f"BM25 results: "
        f"{len(bm25_results)}"
    )

    print(
        "\n=============================="
    )

    print(
        "COMBINING WITH RRF"
    )

    print(
        "=============================="
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        top_k=top_k,
        k=RRF_K,
    )

    print(
        f"Hybrid results: "
        f"{len(hybrid_results)}"
    )

    return hybrid_results


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank_results(
    query,
    results,
    top_k=RERANK_TOP_K,
):

    if not results:
        return []

    model = get_reranker()

    print(
        f"\nReranking "
        f"{len(results)} candidates..."
    )

    pairs = []
    valid_results = []

    for result in results:

        text = get_result_text(result)

        if not text.strip():
            continue

        pairs.append(
            [
                query,
                text,
            ]
        )

        valid_results.append(result)

    if not pairs:
        return []

    scores = model.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for result, score in zip(
        valid_results,
        scores,
    ):

        item = dict(result)

        item[
            "reranker_score"
        ] = float(score)

        reranked.append(item)

    # Highest relevance first.
    reranked.sort(
        key=lambda item: item[
            "reranker_score"
        ],
        reverse=True,
    )

    print(
        "\nCross-encoder scores:"
    )

    for index, item in enumerate(
        reranked,
        start=1,
    ):

        print(
            f"  {index}. "
            f"{item['reranker_score']:.4f}"
        )

    # --------------------------------------------------------
    # Remove weak matches.
    # --------------------------------------------------------

    filtered = [
        item
        for item in reranked
        if item[
            "reranker_score"
        ] >= MIN_RERANK_SCORE
    ]

    print(
        f"\nChunks above relevance threshold: "
        f"{len(filtered)}"
    )

    # --------------------------------------------------------
    # Safety fallback.
    #
    # If every result is below the threshold, keep the
    # strongest result rather than sending no context.
    # --------------------------------------------------------

    if not filtered:

        filtered = reranked[:1]

        print(
            "No chunk passed threshold. "
            "Keeping the best available chunk."
        )

    return filtered[:top_k]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    context_number = 1

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        source = metadata.get(
            "source",
            "Unknown",
        )

        page = metadata.get(
            "page",
            "Unknown",
        )

        human_page = metadata.get(
            "human_page"
        )

        text = get_result_text(result)

        if not text.strip():
            continue

        if human_page is not None:

            page_info = (
                f"Page: {page} "
                f"(Human page: {human_page})"
            )

        else:

            page_info = (
                f"Page: {page}"
            )

        reranker_score = result.get(
            "reranker_score"
        )

        if reranker_score is not None:

            score_info = (
                f"\nRelevance score: "
                f"{reranker_score:.4f}"
            )

        else:

            score_info = ""

        context_parts.append(
            f"""
--- Context {context_number} ---

Source: {source}

{page_info}{score_info}

{text}
"""
        )

        context_number += 1

    return "\n".join(
        context_parts
    )


# ============================================================
# RAG PROMPT
# ============================================================

def create_rag_prompt(
    question,
    context,
):

    return f"""
You are a precise document question-answering assistant.

Your task is to answer the USER QUESTION using ONLY the
DOCUMENT CONTEXT provided below.

STRICT RULES:

1. Answer only from the document context.

2. Do NOT use outside knowledge.

3. Do NOT guess.

4. Do NOT invent facts.

5. Prefer the context section that directly answers
   the question.

6. Do NOT combine unrelated context sections.

7. Only combine multiple context sections when they clearly
   refer to the same topic and are both necessary to answer
   the question.

8. If the context does not contain enough information to
   answer the question, do not guess.

9. Keep the answer concise and directly answer the question.

10. If the answer is not present in the provided context,
    respond EXACTLY:

I could not find the answer in the provided documents.

11. For a successful answer, provide the source and page
    used in this format:

Source: <source>
Page: <page>

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


# ============================================================
# COMPLETE RAG ANSWER
# ============================================================

def rag_answer(
    question,
    bm25,
    chunks,
    return_details=False,
):

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    hybrid_results = create_hybrid_results(
        query=question,
        bm25=bm25,
        chunks=chunks,
        top_k=HYBRID_TOP_K,
    )

    if not hybrid_results:

        answer = (
            "I could not find the answer "
            "in the provided documents."
        )

        if return_details:

            return {
                "answer": answer,
                "retrieved_chunk_ids": [],
                "retrieved_results": [],
                "context": "",
            }

        return answer

    # --------------------------------------------------------
    # Cross encoder reranking
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "RUNNING CROSS-ENCODER RERANKER"
    )

    print(
        "=============================="
    )

    reranked_results = rerank_results(
        query=question,
        results=hybrid_results,
        top_k=RERANK_TOP_K,
    )

    print(
        f"Reranked results: "
        f"{len(reranked_results)}"
    )

    if not reranked_results:

        answer = (
            "I could not find the answer "
            "in the provided documents."
        )

        if return_details:

            return {
                "answer": answer,
                "retrieved_chunk_ids": [],
                "retrieved_results": [],
                "context": "",
            }

        return answer

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "BUILDING CONTEXT"
    )

    print(
        "=============================="
    )

    context = build_context(
        reranked_results
    )

    if not context.strip():

        answer = (
            "I could not find the answer "
            "in the provided documents."
        )

        if return_details:

            return {
                "answer": answer,
                "retrieved_chunk_ids": [],
                "retrieved_results": [],
                "context": "",
            }

        return answer

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = create_rag_prompt(
        question=question,
        context=context,
    )

    # --------------------------------------------------------
    # Groq
    # --------------------------------------------------------

    print(
        "\n=============================="
    )

    print(
        "SENDING CONTEXT TO GROQ"
    )

    print(
        "=============================="
    )

    answer = generate_answer(
        prompt
    )

    # --------------------------------------------------------
    # Retrieved chunk IDs
    # --------------------------------------------------------

    retrieved_chunk_ids = []

    for result in reranked_results:

        chunk_id = result.get(
            "id"
        )

        if chunk_id:

            retrieved_chunk_ids.append(
                str(chunk_id)
            )

    # --------------------------------------------------------
    # Return details for evaluation
    # --------------------------------------------------------

    if return_details:

        return {
            "answer": answer,
            "retrieved_chunk_ids": (
                retrieved_chunk_ids
            ),
            "retrieved_results": (
                reranked_results
            ),
            "context": context,
        }

    return answer


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "             RAG PIPELINE"
    )

    print(
        "========================================"
    )

    print(
        "\nCreating BM25 index..."
    )

    bm25, chunks = (
        create_bm25_index()
    )

    print(
        "\nBM25 index ready."
    )

    print(
        f"Number of chunks: "
        f"{len(chunks)}"
    )

    while True:

        question = input(
            "\nEnter your question "
            "(or 'exit' to quit): "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nExiting RAG pipeline."
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        try:

            answer = rag_answer(
                question=question,
                bm25=bm25,
                chunks=chunks,
            )

            print(
                "\n========================================"
            )

            print(
                "             FINAL RAG ANSWER"
            )

            print(
                "========================================"
            )

            print(answer)

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                type(error).__name__,
                error,
            )
