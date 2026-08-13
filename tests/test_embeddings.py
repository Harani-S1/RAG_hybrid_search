from src.embeddings.embedding import create_embedding_model


def test_embedding_creation():
    embedding_model = create_embedding_model()

    text = "What is machine learning?"
    vector = embedding_model.embed_query(text)

    assert vector is not None
    assert len(vector) > 0