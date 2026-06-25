from pearl.data.schema import Document
from pearl.preprocessing.chunking import chunk_documents


def test_chunking_preserves_overlap_and_metadata():
    doc = Document(
        doc_id="d1",
        title="Test",
        text="alpha beta gamma delta epsilon zeta eta theta iota kappa",
    )
    chunks = chunk_documents([doc], chunk_size=4, overlap=2)

    assert len(chunks) == 4
    assert chunks[0].text == "alpha beta gamma delta"
    assert chunks[1].text == "gamma delta epsilon zeta"
    assert chunks[0].doc_id == "d1"
    assert chunks[0].token_len == 4
