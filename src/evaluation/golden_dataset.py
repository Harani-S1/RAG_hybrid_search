"""
Golden QA Dataset for RAG Evaluation

Phase 4 requirements:
- 50+ question-answer pairs
- Straightforward questions
- Multi-hop questions
- No-answer questions
- Ambiguous questions

The reference answers are based on the project documents:
- Deep_Learning.pdf
- Building_Machine_Learning_Systems.pdf

IMPORTANT:
Relevant chunk IDs are kept only where they are already verified
in the existing project evaluation data. New ground-truth chunk IDs
will be validated/populated during retrieval evaluation.
"""

GOLDEN_DATASET = [

    # ============================================================
    # STRAIGHTFORWARD / LOOKUP QUESTIONS
    # ============================================================

    {
        "question": "What is machine learning?",
        "answer": (
            "Machine learning is a technique that allows computer "
            "systems to improve with experience and data."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0114_000_000975"
        ],
    },

    {
        "question": "What is deep learning?",
        "answer": (
            "Deep learning is a type of machine learning based on "
            "models that involve greater composition of learned "
            "functions or concepts than traditional machine learning."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0028_001_000748"
        ],
    },

    {
        "question": "What is representation learning?",
        "answer": (
            "Representation learning is an approach in which a "
            "machine learning system learns useful representations "
            "or features from data."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0023_001_000738"
        ],
    },

    {
        "question": "What is a neural network?",
        "answer": (
            "A neural network is a machine learning model composed "
            "of interconnected computational units that can learn "
            "representations from data."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0028_000_000747"
        ],
    },

    {
        "question": "What is supervised learning?",
        "answer": (
            "Supervised learning is a type of machine learning in "
            "which the system learns from examples that include "
            "target information or labels."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0120_000_000997"
        ],
    },

    {
        "question": "What is reinforcement learning?",
        "answer": (
            "Reinforcement learning is a type of learning in which "
            "an algorithm interacts with an environment and receives "
            "feedback from its experiences."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0121_001_001001"
        ],
    },

    {
        "question": "What are the main advantages of deep learning?",
        "answer": (
            "Deep learning benefits from more powerful computers, "
            "larger datasets, and techniques that allow deeper "
            "networks to be trained. Deep models can also provide "
            "advantages through distributed representations."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0041_001_000792"
        ],
    },

    {
        "question": "How does deep learning differ from traditional machine learning?",
        "answer": (
            "Deep learning involves a greater amount of composition "
            "of learned functions or learned concepts than traditional "
            "machine learning."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0034_002_000771"
        ],
    },

    {
        "question": "What is a machine learning algorithm?",
        "answer": (
            "A machine learning algorithm is an algorithm that is "
            "able to learn from data."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0114_000_000975"
        ],
    },

    {
        "question": "Why has deep learning become more useful over time?",
        "answer": (
            "Deep learning has become more useful because available "
            "training data has increased and computer infrastructure "
            "has improved. Models have grown in size and have solved "
            "increasingly complicated applications with increasing accuracy."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [
            "chunk_0041_001_000792",
            "chunk_0026_002_000745"
        ],
    },

    {
        "question": "What is overfitting?",
        "answer": (
            "Overfitting occurs when a model fits the training data "
            "too closely and does not generalize well to new examples."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is underfitting?",
        "answer": (
            "Underfitting occurs when a model is not sufficiently "
            "capable of fitting the underlying patterns in the data."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What are hyperparameters?",
        "answer": (
            "Hyperparameters are settings used to control the behavior "
            "of a machine learning algorithm."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is a validation set used for?",
        "answer": (
            "A validation set is used to help select hyperparameters "
            "and evaluate model choices during development."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is stochastic gradient descent?",
        "answer": (
            "Stochastic gradient descent is an optimization approach "
            "that updates model parameters using gradients estimated "
            "from training examples or minibatches."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is classification?",
        "answer": (
            "Classification is a machine learning task in which a "
            "program assigns an input to one of a set of categories."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is regression?",
        "answer": (
            "Regression is a supervised learning task in which the "
            "model learns to predict a continuous-valued output."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is unsupervised learning?",
        "answer": (
            "Unsupervised learning refers to learning from data without "
            "a supervision signal or target provided for each example."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is clustering?",
        "answer": (
            "Clustering consists of dividing a dataset into groups of "
            "similar examples."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is a cost function?",
        "answer": (
            "A cost function measures how well a model performs and "
            "provides an objective that can be optimized during learning."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is maximum likelihood estimation?",
        "answer": (
            "Maximum likelihood estimation is a method for selecting "
            "model parameters by maximizing the likelihood of the "
            "observed training data."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is regularization?",
        "answer": (
            "Regularization introduces constraints or penalties that "
            "discourage overly complex models and can improve generalization."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is dropout?",
        "answer": (
            "Dropout is a regularization technique in which units are "
            "randomly omitted during training."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is early stopping?",
        "answer": (
            "Early stopping is a regularization strategy that stops "
            "training before continued optimization causes poorer "
            "generalization."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is back-propagation?",
        "answer": (
            "Back-propagation is a differentiation algorithm used to "
            "efficiently compute gradients through neural networks."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is a convolutional network?",
        "answer": (
            "A convolutional network is a neural network architecture "
            "that uses convolution operations and is especially useful "
            "for structured data such as images."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is pooling in a convolutional network?",
        "answer": (
            "Pooling is an operation used in convolutional networks "
            "to aggregate information over local regions."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is a recurrent neural network?",
        "answer": (
            "A recurrent neural network is a neural network architecture "
            "designed for processing sequences by maintaining recurrent "
            "connections over time."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What problem do long-term dependencies create for recurrent networks?",
        "answer": (
            "Long-term dependencies make it difficult for recurrent "
            "networks to learn relationships between events that are "
            "far apart in a sequence."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What are LSTM networks?",
        "answer": (
            "LSTM networks are gated recurrent neural networks designed "
            "to help handle long-term dependencies in sequential data."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the purpose of performance metrics?",
        "answer": (
            "Performance metrics provide quantitative measures that can "
            "be used to evaluate how well a machine learning system "
            "performs on its intended task."
        ),
        "type": "lookup",
        "relevant_chunk_ids": [],
    },

    # ============================================================
    # MULTI-HOP QUESTIONS
    # ============================================================

    {
        "question": (
            "Why did improvements in both datasets and computing "
            "infrastructure contribute to the increased usefulness "
            "of deep learning?"
        ),
        "answer": (
            "Deep learning became more useful as larger amounts of "
            "training data became available and computer infrastructure "
            "improved. Better infrastructure allowed larger models to "
            "be trained, while larger datasets provided more information "
            "from which those models could learn."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "How do deep representations help address the challenges "
            "created by high-dimensional data?"
        ),
        "answer": (
            "High-dimensional data creates the curse of dimensionality, "
            "making generalization difficult. Deep learning uses deep, "
            "distributed representations and composition of learned "
            "features to obtain advantages that help overcome these "
            "challenges."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "How are supervised and unsupervised learning different "
            "when considering the information available during training?"
        ),
        "answer": (
            "Supervised learning uses examples that include a target "
            "or label, while unsupervised learning works without such "
            "a supervision signal and attempts to extract information "
            "from the distribution of the data."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "Why can deep learning models benefit more from increasing "
            "amounts of training data?"
        ),
        "answer": (
            "The book describes increasing training data as an important "
            "historical trend in deep learning. Larger datasets provide "
            "more examples from which large models can learn, while "
            "improved infrastructure makes training those larger models "
            "practical."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "How does regularization relate to overfitting?"
        ),
        "answer": (
            "Overfitting occurs when a model does not generalize well "
            "to new examples. Regularization is used to discourage "
            "excessively complex solutions and thereby improve "
            "generalization."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "How do hyperparameters and validation sets work together "
            "during model development?"
        ),
        "answer": (
            "Hyperparameters control aspects of a learning algorithm, "
            "and validation data can be used to compare different "
            "hyperparameter choices and select a model configuration "
            "before final testing."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "How does back-propagation support gradient-based learning "
            "in neural networks?"
        ),
        "answer": (
            "Back-propagation efficiently computes gradients through "
            "the layers of a neural network. Those gradients can then "
            "be used by gradient-based optimization methods to update "
            "the model parameters."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "Why are convolution and pooling useful together in "
            "convolutional neural networks?"
        ),
        "answer": (
            "Convolution extracts local patterns using learned filters, "
            "while pooling aggregates information over local regions. "
            "Together they provide a useful way to process structured "
            "spatial data such as images."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "Why are gated recurrent networks useful for sequence "
            "modeling?"
        ),
        "answer": (
            "Sequence models can struggle with long-term dependencies. "
            "Gated recurrent architectures such as LSTMs introduce "
            "mechanisms that help control the flow of information and "
            "make learning long-term relationships easier."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "How do exploration and exploitation differ in "
            "reinforcement learning?"
        ),
        "answer": (
            "Exploitation means choosing actions based on the current "
            "best policy to obtain a known or expected high reward. "
            "Exploration means trying actions to obtain additional "
            "information and training data. Reinforcement learning "
            "must balance these two goals."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "Why is evaluating reinforcement learning different from "
            "evaluating ordinary supervised learning?"
        ),
        "answer": (
            "In reinforcement learning, the learner interacts with an "
            "environment and its policy influences which inputs and "
            "actions are encountered. This makes evaluation less "
            "straightforward than using a fixed test set with known "
            "target outputs."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "How does the definition of machine learning connect "
            "experience, tasks, and performance?"
        ),
        "answer": (
            "The book describes learning in terms of a program improving "
            "its performance on a class of tasks, as measured by a "
            "performance measure, through experience."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "Why can the distinction between supervised and unsupervised "
            "learning sometimes be blurred?"
        ),
        "answer": (
            "The distinction is not formally rigid. Some machine "
            "learning technologies can be used for both supervised and "
            "unsupervised tasks, and an apparently unsupervised problem "
            "can sometimes be transformed into supervised learning problems."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    {
        "question": (
            "How does the curse of dimensionality motivate the use "
            "of deep learning?"
        ),
        "answer": (
            "As the number of dimensions increases, many machine "
            "learning problems become much harder and generalization "
            "becomes more difficult. Deep learning was developed in "
            "part to overcome these limitations by learning complicated "
            "functions in high-dimensional spaces."
        ),
        "type": "multi_hop",
        "relevant_chunk_ids": [],
    },

    # ============================================================
    # AMBIGUOUS QUESTIONS
    # ============================================================

    {
        "question": "What is the best machine learning algorithm?",
        "answer": (
            "The documents do not identify one universally best machine "
            "learning algorithm. The appropriate algorithm depends on "
            "the task, data, model assumptions, and evaluation criteria."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "How much deep learning is enough?",
        "answer": (
            "The question is ambiguous because the appropriate model "
            "depth depends on the architecture, task, and definition "
            "of depth. The book notes that there is no single consensus "
            "about how much depth is required to qualify a model as deep."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "Which model should I use?",
        "answer": (
            "The question is too ambiguous to answer without knowing "
            "the task, dataset, objective, and constraints."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "Is deep learning always better than machine learning?",
        "answer": (
            "The documents do not support the claim that deep learning "
            "is always better. Deep learning is a specific kind of "
            "machine learning and is particularly useful for some "
            "complex tasks and high-dimensional data."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What does better performance mean?",
        "answer": (
            "The question is ambiguous because performance depends on "
            "the task and the performance metric being used."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is a large dataset?",
        "answer": (
            "The documents discuss increasing dataset sizes as an "
            "important trend, but they do not provide one universal "
            "numeric definition of what constitutes a large dataset."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "Which neural network is the most powerful?",
        "answer": (
            "The documents do not identify one universally most powerful "
            "neural network. Different architectures are suited to "
            "different tasks."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "Why is this method better?",
        "answer": (
            "The question is ambiguous because it does not specify "
            "which method or which performance criterion is being discussed."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "How deep should a network be?",
        "answer": (
            "There is no single answer. The book notes that there is "
            "no single correct value for the depth of an architecture "
            "and no consensus about how much depth is required to "
            "qualify as deep."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the best way to train a model?",
        "answer": (
            "The question is ambiguous because training choices depend "
            "on the model, dataset, objective, optimization method, "
            "and other factors."
        ),
        "type": "ambiguous",
        "relevant_chunk_ids": [],
    },

    # ============================================================
    # NO-ANSWER QUESTIONS
    # ============================================================

    {
        "question": "What is the capital city of France?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "Who is the current president of the United States?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the population of India?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the boiling point of water at sea level?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "Who won the FIFA World Cup in 2022?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the current price of Bitcoin?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the weather in New York today?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the latest version of Python?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "Who is the CEO of Google?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the distance from Earth to the Moon?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the population of Japan?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

    {
        "question": "What is the current stock price of Apple?",
        "answer": (
            "The provided documents do not contain the answer to this question."
        ),
        "type": "no_answer",
        "relevant_chunk_ids": [],
    },

]


# ============================================================
# Dataset statistics
# ============================================================

def get_dataset_statistics():
    """
    Return counts by question type.
    """

    statistics = {
        "total": len(GOLDEN_DATASET),
        "lookup": 0,
        "multi_hop": 0,
        "ambiguous": 0,
        "no_answer": 0,
    }

    for item in GOLDEN_DATASET:

        question_type = item.get(
            "type",
            "unknown",
        )

        if question_type in statistics:
            statistics[question_type] += 1

    return statistics


if __name__ == "__main__":

    stats = get_dataset_statistics()

    print("=" * 60)
    print("GOLDEN QA DATASET")
    print("=" * 60)

    print(
        f"Total questions : {stats['total']}"
    )

    print(
        f"Lookup          : {stats['lookup']}"
    )

    print(
        f"Multi-hop       : {stats['multi_hop']}"
    )

    print(
        f"Ambiguous       : {stats['ambiguous']}"
    )

    print(
        f"No-answer       : {stats['no_answer']}"
    )

    print("=" * 60)