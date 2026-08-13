from langchain_community.document_loaders import PyPDFLoader
from src.chunking.chunker import chunk_documents


def test_chunking():
    pdf_path = "data/raw/Deep_Learning.pdf"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    chunks = chunk_documents(
        documents,
        strategy="fixed",
        chunk_size=1000,
        chunk_overlap=200,
    )

    assert len(documents) > 0
    assert len(chunks) > 0

    for chunk in chunks[:5]:
        assert chunk.page_content.strip() != ""
        assert "page" in chunk.metadata