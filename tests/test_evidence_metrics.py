from pearl.data.schema import Chunk, EvidenceSpan
from pearl.evaluation.evidence_metrics import evidence_f1, evidence_precision, evidence_recall


def test_evidence_overlap_metrics():
    selected = [
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            text="contains evidence",
            token_len=2,
            section_title=None,
            chunk_index=0,
            char_start=10,
            char_end=20,
        )
    ]
    gold = [EvidenceSpan(doc_id="d1", char_start=15, char_end=25)]

    assert evidence_recall(selected, gold, threshold=0.3) == 1.0
    assert evidence_precision(selected, gold, threshold=0.3) == 1.0
    assert evidence_f1(selected, gold, threshold=0.3) == 1.0
