"""
Automated RAG Evaluation
"""

from src.evaluation.golden_dataset import GOLDEN_DATASET
from src.retrieval.bm25_retriever import create_bm25_index
from src.generation.rag_pipeline import rag_answer


def run_evaluation():

    print("=" * 60)
    print("AUTOMATED RAG EVALUATION")
    print("=" * 60)

    print("\nCreating BM25 index...")

    bm25, chunks = create_bm25_index()

    print("\nBM25 index ready.")
    print(f"Total chunks: {len(chunks)}")
    print(f"Golden questions: {len(GOLDEN_DATASET)}")

    results = []

    for index, item in enumerate(
        GOLDEN_DATASET,
        start=1,
    ):

        question = item["question"]

        print()
        print("=" * 60)
        print(
            f"QUESTION {index}/{len(GOLDEN_DATASET)}"
        )
        print("=" * 60)

        print(f"\nQuestion: {question}")

        try:
            answer = rag_answer(
                question,
                bm25,
                chunks,
            )

        except Exception as error:

            print("\nRAG ERROR:")
            print(error)

            answer = "ERROR: RAG pipeline failed."

        print("\nGenerated answer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)

        results.append(
            {
                "question": question,
                "expected_answer": item["answer"],
                "generated_answer": answer,
                "type": item["type"],
                "relevant_chunk_ids": item[
                    "relevant_chunk_ids"
                ],
            }
        )

    print()
    print("=" * 60)
    print("EVALUATION RUN COMPLETED")
    print("=" * 60)

    print(
        f"\nQuestions processed: {len(results)}"
    )


if __name__ == "__main__":
    run_evaluation()