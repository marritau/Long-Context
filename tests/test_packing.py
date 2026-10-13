from bikv.data import generate_dataset
from bikv.packing import FixedBlockPacker


class DummyTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(character) % 89) for character in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        rendered = "".join(f"<{row['role']}>\n{row['content']}\n" for row in messages)
        return rendered + ("<assistant>\n" if add_generation_prompt else "")


def test_packer_produces_exact_equal_block_costs():
    sample = generate_dataset(n_stories=1, n_blocks=10, seed=3)[0]
    packer = FixedBlockPacker(DummyTokenizer(), block_size=512, use_chat_template=False)
    packed = packer.pack(sample)
    layout = packed.layout
    assert layout.data_end - layout.data_start == 10 * 512
    assert layout.original_length == packed.context_ids.shape[1]
    assert layout.protected_tokens + 10 * 512 == layout.original_length
    assert layout.suffix_length == 0
    assert packed.query_ids.shape[1] > 0
    assert len(layout.block_content_lengths) == 10
    assert all(0 < length < layout.block_size for length in layout.block_content_lengths)


def test_packer_refuses_to_truncate_evidence():
    sample = generate_dataset(n_stories=1, n_blocks=10, seed=3)[0]
    packer = FixedBlockPacker(DummyTokenizer(), block_size=5, use_chat_template=False)
    try:
        packer.pack(sample)
    except ValueError as error:
        assert "increase block_size" in str(error)
    else:
        raise AssertionError("Expected evidence truncation guard")


def test_chat_template_preserves_exact_document_layout_and_overrides():
    sample = generate_dataset(n_stories=1, n_blocks=10, seed=3)[0]
    packer = FixedBlockPacker(DummyTokenizer(), block_size=512, use_chat_template=True)
    packed = packer.pack(
        sample,
        question="Which technician is assigned to Orion?",
        answer="Mira",
        answer_aliases=["Mira"],
    )
    assert packed.layout.data_start == packed.layout.prefix_length
    assert packed.layout.data_end == packed.layout.original_length
    assert packed.layout.suffix_length == 0
    assert packed.query_ids.shape[1] > 0
    assert packed.layout.data_end - packed.layout.data_start == 10 * 512
    assert packed.answer_text == "Mira"
    assert packed.answer_aliases == ["Mira"]
