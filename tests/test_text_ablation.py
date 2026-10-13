from bikv.runner import _pack_text_ablation
from bikv.schema import StorySample


class CaptureBackend:
    def __init__(self):
        self.sample = None

    def pack(self, sample):
        self.sample = sample
        return sample


def test_text_ablation_removes_evidence_before_packing():
    sample = StorySample(
        sample_id="sample",
        story_id="story",
        split="train",
        variant="neutral",
        blocks=["secret left fact", "secret right fact", "innocent filler"],
        question="Question?",
        answer="AA-001",
        answer_aliases=["AA-001"],
        supporting_blocks=[0, 1],
        manipulated_boundary=0,
    )
    backend = CaptureBackend()

    packed = _pack_text_ablation(backend, sample, {1})

    assert packed.blocks[0] == "secret left fact"
    assert "secret right fact" not in " ".join(packed.blocks)
    assert "No assignment or verification information" in packed.blocks[1]
    assert sample.blocks[1] == "secret right fact"

