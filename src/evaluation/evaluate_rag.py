"""
Automated RAG Evaluation
Phase 4 Metrics

Evaluation types:
- lookup
- multi_hop
- ambiguous
- no_answer

Important:
This file evaluates the RAG system.
It does NOT change the retrieval or generation model itself.

The evaluation logic is question-type aware so that:
- lookup questions are evaluated for answer correctness
- multi-hop questions are evaluated for answer correctness
- ambiguous questions are evaluated for appropriate ambiguity handling
- no-answer questions are evaluated for correct abstention
- missing retrieval ground truth is NOT treated as retrieval failure
- citations are not required for no-answer responses
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.retrieval.bm25_retriever import create_bm25_index
from src.generation.rag_pipeline import rag_answer
from src.evaluation.golden_dataset import GOLDEN_DATASET


# ============================================================
# CONFIGURATION
# ============================================================

CORRECTNESS_THRESHOLD = 0.50


# ============================================================
# NO-ANSWER PHRASES
# ============================================================

NO_ANSWER_PHRASES = [
    "i could not find the answer",
    "could not find the answer",
    "couldn't find the answer",
    "not found in the provided documents",
    "not present in the provided documents",
    "documents do not contain the answer",
    "the documents do not contain the answer",
    "information is not available in the provided documents",
    "not enough information",
    "the provided documents do not contain",
    "not available in the provided documents",
    "cannot be answered from the provided documents",
    "cannot answer from the provided documents",
    "the documents do not provide",
]


# ============================================================
# AMBIGUITY PHRASES
# ============================================================

AMBIGUITY_PHRASES = [
    "ambiguous",
    "too ambiguous",
    "not enough information",
    "depends on",
    "depends upon",
    "cannot determine",
    "cannot be determined",
    "no single answer",
    "no universal answer",
    "not enough context",
    "need more information",
    "without knowing",
    "depends on the task",
    "depends on the dataset",
    "depends on the model",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def remove_citations(text):
    """
    Remove citation lines before lexical evaluation.
    """

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"(?im)^\s*source\s*:\s*.*$",
        "",
        text,
    )

    text = re.sub(
        r"(?im)^\s*page\s*:\s*.*$",
        "",
        text,
    )

    return text


def normalize_text(text):
    """
    Normalize text for evaluation.
    """

    if not text:
        return ""

    text = remove_citations(text)

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def get_words(text):
    """
    Return unique normalized words.
    """

    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(normalized.split())


# ============================================================
# NO-ANSWER DETECTION
# ============================================================

def is_no_answer_response(answer):
    """
    Detect whether the generated answer correctly indicates
    that the information is unavailable.
    """

    if not answer:
        return False

    normalized = normalize_text(answer)

    for phrase in NO_ANSWER_PHRASES:

        if phrase in normalized:
            return True

    return False


# ============================================================
# AMBIGUOUS RESPONSE DETECTION
# ============================================================

def is_ambiguous_response(answer):
    """
    Detect whether the generated response appropriately
    recognizes an ambiguous question.
    """

    if not answer:
        return False

    normalized = normalize_text(answer)

    for phrase in AMBIGUITY_PHRASES:

        if phrase in normalized:
            return True

    return False


# ============================================================
# ANSWER CORRECTNESS
# ============================================================

def lexical_answer_score(
    generated_answer,
    expected_answer,
):
    """
    Calculate token-level F1 for normal answerable questions.
    """

    generated_words = get_words(
        generated_answer
    )

    expected_words = get_words(
        expected_answer
    )

    if not expected_words:
        return 0.0

    if not generated_words:
        return 0.0

    matched = (
        generated_words
        & expected_words
    )

    precision = (
        len(matched)
        / len(generated_words)
    )

    recall = (
        len(matched)
        / len(expected_words)
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def evaluate_answer_correctness(
    generated_answer,
    expected_answer,
    question_type,
):
    """
    Evaluate answer correctness according to question type.

    lookup:
        lexical F1

    multi_hop:
        lexical F1

    no_answer:
        correct abstention = 1.0

    ambiguous:
        appropriate ambiguity handling = 1.0
    """

    # --------------------------------------------------------
    # NO-ANSWER
    # --------------------------------------------------------

    if question_type == "no_answer":

        if is_no_answer_response(
            generated_answer
        ):
            return 1.0

        return 0.0

    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

    if question_type == "ambiguous":

        if is_ambiguous_response(
            generated_answer
        ):
            return 1.0

        return 0.0

    # --------------------------------------------------------
    # LOOKUP / MULTI-HOP
    # --------------------------------------------------------

    return lexical_answer_score(
        generated_answer,
        expected_answer,
    )


# ============================================================
# FAITHFULNESS
# ============================================================

def evaluate_faithfulness(
    generated_answer,
    context,
    question_type,
):
    """
    Estimate lexical support of generated content.

    For no-answer responses there is no factual answer to
    support, so they are not unfairly penalized.

    For ambiguous responses, the ambiguity handling itself
    does not require retrieved factual support.
    """

    if question_type in {
        "no_answer",
        "ambiguous",
    }:

        return 1.0

    answer_words = get_words(
        generated_answer
    )

    context_words = get_words(
        context
    )

    if not answer_words:
        return 0.0

    if not context_words:
        return 0.0

    supported = (
        answer_words
        & context_words
    )

    return (
        len(supported)
        / len(answer_words)
    )


# ============================================================
# RETRIEVAL RELEVANCE
# ============================================================

def evaluate_retrieval_relevance(
    retrieved_chunk_ids,
    relevant_chunk_ids,
):
    """
    Calculate retrieval recall when ground-truth chunk IDs
    are actually available.

    Empty relevant_chunk_ids means:
    ground truth has not been populated.

    Therefore it returns None rather than 0.
    """

    if not relevant_chunk_ids:
        return None

    retrieved = set(
        str(x)
        for x in retrieved_chunk_ids
        if x is not None
    )

    relevant = set(
        str(x)
        for x in relevant_chunk_ids
        if x is not None
    )

    if not relevant:
        return None

    matched = (
        retrieved
        & relevant
    )

    return (
        len(matched)
        / len(relevant)
    )


# ============================================================
# CITATION PARSING
# ============================================================

def extract_citations(answer):
    """
    Extract:

    Source: filename
    Page: page_number
    """

    if not answer:
        return []

    pattern = re.compile(
        r"source\s*:\s*(.+?)"
        r"(?:\n|\r\n)"
        r"\s*page\s*:\s*([^\n\r]+)",
        flags=re.IGNORECASE,
    )

    matches = pattern.findall(
        str(answer)
    )

    citations = []

    for source, page in matches:

        source = source.strip()
        page = page.strip()

        if source:

            citations.append(
                {
                    "source": source,
                    "page": page,
                }
            )

    unique = []

    seen = set()

    for citation in citations:

        key = (
            citation["source"].lower(),
            citation["page"].lower(),
        )

        if key not in seen:

            seen.add(key)

            unique.append(
                citation
            )

    return unique


# ============================================================
# CITATION NORMALIZATION
# ============================================================

def normalize_source(source):
    """
    Normalize source filename.
    """

    if not source:
        return ""

    source = str(
        source
    ).strip().lower()

    source = source.replace(
        "\\",
        "/",
    )

    return source.split("/")[-1]


def normalize_page(page):
    """
    Normalize page number.
    """

    if page is None:
        return ""

    page = str(
        page
    ).strip().lower()

    page = re.sub(
        r"^page\s*[:\-]?\s*",
        "",
        page,
    )

    match = re.search(
        r"\d+",
        page,
    )

    if match:
        return match.group(0)

    return page


# ============================================================
# CITATION ACCURACY
# ============================================================

def evaluate_citation_accuracy(
    generated_answer,
    retrieved_results,
    question_type,
):
    """
    Evaluate citation correctness.

    No-answer and ambiguous questions are not required to
    provide factual source/page citations.
    """

    if question_type in {
        "no_answer",
        "ambiguous",
    }:

        return 1.0

    citations = extract_citations(
        generated_answer
    )

    if not citations:
        return 0.0

    if not retrieved_results:
        return 0.0

    valid_references = []

    for result in retrieved_results:

        if not isinstance(
            result,
            dict,
        ):
            continue

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
            "",
        )

        page = metadata.get(
            "page",
            "",
        )

        if source:

            valid_references.append(
                {
                    "source": normalize_source(
                        source
                    ),
                    "page": normalize_page(
                        page
                    ),
                }
            )

    if not valid_references:
        return 0.0

    correct = 0

    for citation in citations:

        cited_source = normalize_source(
            citation["source"]
        )

        cited_page = normalize_page(
            citation["page"]
        )

        for reference in valid_references:

            source_matches = (
                cited_source
                == reference["source"]
            )

            page_matches = (
                not cited_page
                or not reference["page"]
                or cited_page
                == reference["page"]
            )

            if (
                source_matches
                and page_matches
            ):

                correct += 1

                break

    return min(
        correct / len(citations),
        1.0,
    )


# ============================================================
# OVERALL METRIC
# ============================================================

def calculate_overall_score(
    answer_correctness,
    faithfulness,
    retrieval_relevance,
    citation_accuracy,
):
    """
    Calculate overall metric score.

    Retrieval relevance is included only when ground-truth
    chunk IDs are available.
    """

    scores = [
        answer_correctness,
        faithfulness,
        citation_accuracy,
    ]

    if retrieval_relevance is not None:

        scores.append(
            retrieval_relevance
        )

    return (
        sum(scores)
        / len(scores)
        if scores
        else 0.0
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    summary,
):
    """
    Save evaluation results to JSON.
    """

    output_directory = (
        PROJECT_ROOT
        / "evaluation_results"
    )

    output_directory.mkdir(
        exist_ok=True
    )

    output_file = (
        output_directory
        / "rag_evaluation_results.json"
    )

    data = {
        "evaluation_type": "automated",
        "metrics": [
            "answer_correctness",
            "faithfulness",
            "retrieval_relevance",
            "citation_accuracy",
        ],
        "summary": summary,
        "results": results,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_file


# ============================================================
# AVERAGE
# ============================================================

def average(values):
    """
    Average only numeric values.
    """

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if not valid_values:
        return 0.0

    return (
        sum(valid_values)
        / len(valid_values)
    )


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

    print(
        "\n========================================"
    )

    print(
        "       AUTOMATED RAG EVALUATION"
    )

    print(
        "========================================"
    )

    total_questions = len(
        GOLDEN_DATASET
    )

    print(
        f"\nGolden dataset questions: "
        f"{total_questions}"
    )

    # --------------------------------------------------------
    # Question statistics
    # --------------------------------------------------------

    type_counts = Counter(
        item.get(
            "type",
            "unknown",
        )
        for item in GOLDEN_DATASET
    )

    print(
        "\nQuestion types:"
    )

    for question_type, count in (
        type_counts.items()
    ):

        print(
            f"  {question_type}: {count}"
        )

    # --------------------------------------------------------
    # Retrieval ground truth
    # --------------------------------------------------------

    questions_with_retrieval_gt = 0
    questions_without_retrieval_gt = 0

    for item in GOLDEN_DATASET:

        relevant_ids = item.get(
            "relevant_chunk_ids",
            [],
        )

        if relevant_ids:

            questions_with_retrieval_gt += 1

        else:

            questions_without_retrieval_gt += 1

    print(
        "\nRetrieval ground truth:"
    )

    print(
        f"  With relevant chunk IDs    : "
        f"{questions_with_retrieval_gt}"
    )

    print(
        f"  Without relevant chunk IDs : "
        f"{questions_without_retrieval_gt}"
    )

    print(
        "\nMissing retrieval ground truth "
        "is treated as N/A, not failure."
    )

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

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
        f"Number of chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Metric storage
    # --------------------------------------------------------

    correctness_scores = []
    faithfulness_scores = []
    retrieval_scores = []
    citation_scores = []
    overall_scores = []

    results = []

    correct = 0
    incorrect = 0

    # ========================================================
    # QUESTIONS
    # ========================================================

    for index, item in enumerate(
        GOLDEN_DATASET,
        start=1,
    ):

        question = item.get(
            "question",
            "",
        )

        expected_answer = item.get(
            "answer",
            "",
        )

        question_type = item.get(
            "type",
            "lookup",
        )

        relevant_chunk_ids = item.get(
            "relevant_chunk_ids",
            [],
        )

        print(
            "\n========================================"
        )

        print(
            f"QUESTION "
            f"{index}/{total_questions}"
        )

        print(
            "========================================"
        )

        print(
            f"\nType: {question_type}"
        )

        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # Run RAG
        # ----------------------------------------------------

        try:

            rag_result = rag_answer(
                question=question,
                bm25=bm25,
                chunks=chunks,
                return_details=True,
            )

            generated_answer = rag_result.get(
                "answer",
                "",
            )

            retrieved_chunk_ids = (
                rag_result.get(
                    "retrieved_chunk_ids",
                    [],
                )
            )

            retrieved_results = (
                rag_result.get(
                    "retrieved_results",
                    [],
                )
            )

            context = rag_result.get(
                "context",
                "",
            )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                type(error).__name__,
                error,
            )

            generated_answer = (
                "ERROR: RAG pipeline failed."
            )

            retrieved_chunk_ids = []
            retrieved_results = []
            context = ""

        # ----------------------------------------------------
        # Print generated answer
        # ----------------------------------------------------

        print(
            "\n----------------------------------------"
        )

        print(
            "GENERATED ANSWER"
        )

        print(
            "----------------------------------------"
        )

        print(
            generated_answer
        )

        # ====================================================
        # METRIC 1 — ANSWER CORRECTNESS
        # ====================================================

        answer_correctness = (
            evaluate_answer_correctness(
                generated_answer,
                expected_answer,
                question_type,
            )
        )

        # ====================================================
        # METRIC 2 — FAITHFULNESS
        # ====================================================

        faithfulness = (
            evaluate_faithfulness(
                generated_answer,
                context,
                question_type,
            )
        )

        # ====================================================
        # METRIC 3 — RETRIEVAL RELEVANCE
        # ====================================================

        retrieval_relevance = (
            evaluate_retrieval_relevance(
                retrieved_chunk_ids,
                relevant_chunk_ids,
            )
        )

        # ====================================================
        # METRIC 4 — CITATION ACCURACY
        # ====================================================

        citation_accuracy = (
            evaluate_citation_accuracy(
                generated_answer,
                retrieved_results,
                question_type,
            )
        )

        # ====================================================
        # OVERALL
        # ====================================================

        overall_score = (
            calculate_overall_score(
                answer_correctness,
                faithfulness,
                retrieval_relevance,
                citation_accuracy,
            )
        )

        # ----------------------------------------------------
        # Store scores
        # ----------------------------------------------------

        correctness_scores.append(
            answer_correctness
        )

        faithfulness_scores.append(
            faithfulness
        )

        retrieval_scores.append(
            retrieval_relevance
        )

        citation_scores.append(
            citation_accuracy
        )

        overall_scores.append(
            overall_score
        )

        # ----------------------------------------------------
        # Correct / incorrect
        # ----------------------------------------------------

        if (
            answer_correctness
            >= CORRECTNESS_THRESHOLD
        ):

            evaluation = "correct"

            correct += 1

        else:

            evaluation = "incorrect"

            incorrect += 1

        # ----------------------------------------------------
        # Citations
        # ----------------------------------------------------

        citations = extract_citations(
            generated_answer
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            "\n----------------------------------------"
        )

        print(
            "AUTOMATED METRICS"
        )

        print(
            "----------------------------------------"
        )

        print(
            f"Answer Correctness   : "
            f"{answer_correctness * 100:.2f}%"
        )

        print(
            f"Faithfulness         : "
            f"{faithfulness * 100:.2f}%"
        )

        if retrieval_relevance is None:

            print(
                "Retrieval Relevance  : N/A"
            )

        else:

            print(
                f"Retrieval Relevance  : "
                f"{retrieval_relevance * 100:.2f}%"
            )

        print(
            f"Citation Accuracy    : "
            f"{citation_accuracy * 100:.2f}%"
        )

        print(
            f"Overall Metric Score : "
            f"{overall_score * 100:.2f}%"
        )

        print(
            f"Evaluation           : "
            f"{evaluation.upper()}"
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            {
                "question": question,
                "type": question_type,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,

                "answer_correctness": round(
                    answer_correctness,
                    4,
                ),

                "faithfulness": round(
                    faithfulness,
                    4,
                ),

                "retrieval_relevance": (
                    round(
                        retrieval_relevance,
                        4,
                    )
                    if retrieval_relevance
                    is not None
                    else None
                ),

                "citation_accuracy": round(
                    citation_accuracy,
                    4,
                ),

                "overall_score": round(
                    overall_score,
                    4,
                ),

                "evaluation": evaluation,

                "citations": citations,

                "relevant_chunk_ids": (
                    relevant_chunk_ids
                ),

                "retrieved_chunk_ids": (
                    retrieved_chunk_ids
                ),
            }
        )

    # ========================================================
    # AVERAGES
    # ========================================================

    avg_correctness = average(
        correctness_scores
    )

    avg_faithfulness = average(
        faithfulness_scores
    )

    avg_retrieval = average(
        retrieval_scores
    )

    avg_citation = average(
        citation_scores
    )

    avg_overall = average(
        overall_scores
    )

    # ========================================================
    # ACCURACY
    # ========================================================

    overall_accuracy = (
        correct
        / total_questions
        * 100
        if total_questions
        else 0.0
    )

    # --------------------------------------------------------
    # Per-type accuracy
    # --------------------------------------------------------

    type_accuracy = {}

    for question_type in type_counts:

        type_results = [
            result
            for result in results
            if result["type"] == question_type
        ]

        if not type_results:

            continue

        type_correct = sum(
            1
            for result in type_results
            if result["evaluation"]
            == "correct"
        )

        type_accuracy[question_type] = round(
            type_correct
            / len(type_results),
            4,
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "total_questions": (
            total_questions
        ),

        "correct": correct,

        "incorrect": incorrect,

        "overall_accuracy": round(
            overall_accuracy / 100,
            4,
        ),

        "answer_accuracy_percent": round(
            overall_accuracy,
            2,
        ),

        "accuracy_by_type": type_accuracy,

        "average_answer_correctness": round(
            avg_correctness,
            4,
        ),

        "average_faithfulness": round(
            avg_faithfulness,
            4,
        ),

        "average_retrieval_relevance": round(
            avg_retrieval,
            4,
        ),

        "average_citation_accuracy": round(
            avg_citation,
            4,
        ),

        "average_overall_score": round(
            avg_overall,
            4,
        ),

        "retrieval_ground_truth_questions": (
            questions_with_retrieval_gt
        ),

        "retrieval_ground_truth_missing": (
            questions_without_retrieval_gt
        ),
    }

    # ========================================================
    # SAVE
    # ========================================================

    output_file = save_results(
        results,
        summary,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n\n========================================"
    )

    print(
        "     AUTOMATED EVALUATION RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"\nTotal questions      : "
        f"{total_questions}"
    )

    print(
        f"Correct              : "
        f"{correct}"
    )

    print(
        f"Incorrect            : "
        f"{incorrect}"
    )

    print(
        f"\nOverall Accuracy     : "
        f"{overall_accuracy:.2f}%"
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "ACCURACY BY QUESTION TYPE"
    )

    print(
        "----------------------------------------"
    )

    for question_type, accuracy in (
        type_accuracy.items()
    ):

        print(
            f"{question_type:20s}: "
            f"{accuracy * 100:.2f}%"
        )

    print(
        "\n----------------------------------------"
    )

    print(
        "PHASE 4 METRICS"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Answer Correctness   : "
        f"{avg_correctness * 100:.2f}%"
    )

    print(
        f"Faithfulness         : "
        f"{avg_faithfulness * 100:.2f}%"
    )

    if questions_with_retrieval_gt > 0:

        print(
            f"Retrieval Relevance  : "
            f"{avg_retrieval * 100:.2f}%"
        )

    else:

        print(
            "Retrieval Relevance  : N/A"
        )

    print(
        f"Citation Accuracy    : "
        f"{avg_citation * 100:.2f}%"
    )

    print(
        f"Overall Metric Score : "
        f"{avg_overall * 100:.2f}%"
    )

    print(
        "\n----------------------------------------"
    )

    print(
        "RETRIEVAL GROUND TRUTH"
    )

    print(
        "----------------------------------------"
    )

    print(
        f"Questions with chunk IDs : "
        f"{questions_with_retrieval_gt}"
    )

    print(
        f"Questions without IDs    : "
        f"{questions_without_retrieval_gt}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_file
    )

    print(
        "\n========================================"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_evaluation()
