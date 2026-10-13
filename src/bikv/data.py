from __future__ import annotations

import math
import random
import string

from bikv.schema import StorySample


AGENTS = ["Mira", "Daria", "Nolan", "Iris", "Pavel", "Selene", "Tomas", "Vera"]
LABS = ["Orion", "Lumen", "Cedar", "Atlas", "Vega", "Quartz", "Helix", "Nova"]
BOUNDARY_CUE = "NEW RECORD SEGMENT."


def _record_context(story_index: int, slot: int) -> str:
    shifts = ["morning", "afternoon", "evening"]
    archives = ["North", "Central", "South", "West"]
    return (
        f" This entry was copied from the {archives[(story_index + slot) % len(archives)]} archive "
        f"during the {shifts[(story_index * 2 + slot) % len(shifts)]} review. "
        "The timestamp, storage shelf, and reviewer initials are administrative metadata; "
        "they do not change personnel assignments or verification credentials."
    )


def _code(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(rng.choice(letters) for _ in range(2)) + "-" + "".join(
        rng.choice(string.digits) for _ in range(3)
    )


def _split_for_group(index: int, total: int) -> str:
    train_end = max(1, int(total * 0.6))
    validation_end = max(train_end + 1, int(total * 0.8)) if total >= 3 else train_end
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def _choose_sham_boundary(n_blocks: int, support_left: int) -> int:
    forbidden = {support_left - 1, support_left, support_left + 1}
    for boundary in range(n_blocks - 1):
        if boundary not in forbidden:
            return boundary
    raise ValueError("Not enough blocks to create a sham boundary")


def _family_entities(family_index: int, seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed + family_index * 7919)
    agents = rng.sample(AGENTS, 6)
    labs = rng.sample(LABS, 3)
    codes: list[str] = []
    while len(codes) < 6:
        candidate = _code(rng)
        if candidate not in codes:
            codes.append(candidate)
    return agents, labs, codes


def _base_blocks(
    story_index: int,
    n_blocks: int,
    seed: int,
) -> tuple[list[str], int, list[int], dict[str, object]]:
    family_index = story_index // 2
    counterfactual_index = story_index % 2
    agents, labs, codes = _family_entities(family_index, seed)

    # Paired stories reuse entities and codes but rotate the laboratory roster.
    # Cycling by story index balances the gold answer across all six candidates.
    assignment_shift = story_index % 6
    lab_to_agent = [agents[(index + assignment_shift) % 6] for index in range(3)]
    target_lab = labs[0]
    target_agent = lab_to_agent[0]
    target_agent_index = agents.index(target_agent)
    answer = codes[target_agent_index]

    code_records = [
        f"Credential register. The verification code assigned to technician {agent} is {code}."
        + _record_context(story_index, index)
        for index, (agent, code) in enumerate(zip(agents, codes))
    ]
    lab_records = [
        f"Laboratory roster. The technician assigned to laboratory {lab} is {agent}."
        + _record_context(story_index, index + len(code_records))
        for index, (lab, agent) in enumerate(zip(labs, lab_to_agent))
    ]
    target_code = code_records[target_agent_index]
    target_lab_record = lab_records[0]
    remaining = [record for record in code_records + lab_records if record not in {target_code, target_lab_record}]

    rng = random.Random(seed + story_index * 1009)
    rng.shuffle(remaining)
    reserved_codes = set(codes)
    while len(remaining) < n_blocks - 2:
        agent = rng.choice(agents)
        lab = rng.choice(labs[1:])
        audit_code = _code(rng)
        while audit_code in reserved_codes:
            audit_code = _code(rng)
        reserved_codes.add(audit_code)
        remaining.append(
            f"Audit note. Technician {agent} reviewed laboratory {lab}; audit reference {audit_code} was unrelated."
            + _record_context(story_index, len(remaining))
        )
    remaining = remaining[: n_blocks - 2]

    support_left = 1 + (story_index % (n_blocks - 2))
    blocks: list[str | None] = [None] * n_blocks
    evidence_order = "code_then_roster" if story_index % 2 == 0 else "roster_then_code"
    ordered_support = (
        [target_code, target_lab_record]
        if evidence_order == "code_then_roster"
        else [target_lab_record, target_code]
    )
    blocks[support_left], blocks[support_left + 1] = ordered_support
    remaining_iter = iter(remaining)
    for index in range(n_blocks):
        if blocks[index] is None:
            blocks[index] = next(remaining_iter)

    code_by_agent = dict(zip(agents, codes))
    agents_fixed_by_decoy_labs = set(lab_to_agent[1:])
    agents_possible_without_target_lab = [
        agent for agent in agents if agent not in agents_fixed_by_decoy_labs
    ]
    without_code = ["UNOBSERVED_CODE_A", "UNOBSERVED_CODE_B", "UNOBSERVED_CODE_C"]
    without_roster = [code_by_agent[agent] for agent in agents_possible_without_target_lab]
    symbolic_candidates = {
        "without_left": without_code if evidence_order == "code_then_roster" else without_roster,
        "without_right": without_roster if evidence_order == "code_then_roster" else without_code,
        # Both the roster edge and the target code are absent.
        "without_both": [
            code_by_agent[agent]
            for agent in agents_possible_without_target_lab
            if code_by_agent[agent] != answer
        ]
        + ["UNOBSERVED_CODE_A", "UNOBSERVED_CODE_B", "UNOBSERVED_CODE_C"],
    }
    metadata: dict[str, object] = {
        "generator_version": 6,
        "counterfactual_group": f"cf-{family_index:04d}",
        "counterfactual_index": counterfactual_index,
        "candidate_agents": agents,
        "candidate_labs": labs,
        "candidate_codes": codes,
        "symbolic_ablation_candidates": symbolic_candidates,
        "target_agent": target_agent,
        "target_lab": target_lab,
        "support_boundary": support_left,
        "evidence_order": evidence_order,
        "code_block": support_left if evidence_order == "code_then_roster" else support_left + 1,
        "roster_block": support_left + 1 if evidence_order == "code_then_roster" else support_left,
        "generator_seed": seed + story_index * 1009,
    }
    return [str(block) for block in blocks], support_left, [support_left, support_left + 1], {
        **metadata,
        "answer": answer,
    }


def generate_story_variants(story_index: int, n_stories: int, n_blocks: int, seed: int) -> list[StorySample]:
    if n_blocks < 9:
        raise ValueError("At least nine blocks are required for six codes and three laboratory rosters")
    family_count = math.ceil(n_stories / 2)
    family_index = story_index // 2
    story_id = f"story-{story_index:04d}"
    split = _split_for_group(family_index, family_count)
    base, support_left, supporting_blocks, metadata = _base_blocks(story_index, n_blocks, seed)
    answer = str(metadata.pop("answer"))
    target_lab = str(metadata["target_lab"])
    sham_boundary = _choose_sham_boundary(n_blocks, support_left)
    metadata["sham_boundary"] = sham_boundary
    question = (
        f"First identify the technician assigned to laboratory {target_lab}. "
        "Then report the verification code assigned to that technician."
    )

    samples = []
    for variant in ("neutral", "targeted", "sham"):
        blocks = list(base)
        manipulated = support_left if variant != "sham" else sham_boundary
        if variant == "targeted":
            blocks[support_left + 1] = BOUNDARY_CUE + " " + blocks[support_left + 1]
        elif variant == "sham":
            blocks[sham_boundary + 1] = BOUNDARY_CUE + " " + blocks[sham_boundary + 1]

        samples.append(
            StorySample(
                sample_id=f"{story_id}-{variant}",
                story_id=story_id,
                split=split,
                variant=variant,
                blocks=blocks,
                question=question,
                answer=answer,
                answer_aliases=[answer, answer.lower(), answer.replace("-", "")],
                supporting_blocks=supporting_blocks,
                manipulated_boundary=manipulated,
                metadata=dict(metadata),
            )
        )
    return samples


def generate_dataset(n_stories: int, n_blocks: int, seed: int) -> list[StorySample]:
    samples = []
    for story_index in range(n_stories):
        samples.extend(generate_story_variants(story_index, n_stories, n_blocks, seed))
    validate_variant_invariants(samples)
    return samples


def validate_variant_invariants(samples: list[StorySample]) -> None:
    grouped: dict[str, list[StorySample]] = {}
    for sample in samples:
        sample.validate()
        grouped.setdefault(sample.story_id, []).append(sample)
    for story_id, variants in grouped.items():
        if {sample.variant for sample in variants} != {"neutral", "targeted", "sham"}:
            raise ValueError(f"{story_id} does not have all three variants")
        reference = variants[0]
        for sample in variants[1:]:
            if (sample.answer, sample.question, sample.supporting_blocks, sample.split) != (
                reference.answer,
                reference.question,
                reference.supporting_blocks,
                reference.split,
            ):
                raise ValueError(f"Latent facts changed across variants of {story_id}")

    counterfactuals: dict[str, list[StorySample]] = {}
    for sample in samples:
        if sample.variant == "neutral":
            counterfactuals.setdefault(str(sample.metadata["counterfactual_group"]), []).append(sample)
    for family, stories in counterfactuals.items():
        if len({sample.split for sample in stories}) != 1:
            raise ValueError(f"Counterfactual family {family} crosses data splits")
        if len(stories) == 2 and len({sample.answer for sample in stories}) != 2:
            raise ValueError(f"Counterfactual family {family} must change the answer")

    for sample in samples:
        candidates = sample.metadata.get("symbolic_ablation_candidates", {})
        for ablation in ("without_left", "without_right", "without_both"):
            if len(set(candidates.get(ablation, []))) < 3:
                raise ValueError(
                    f"{sample.sample_id} is logically recoverable after {ablation}; "
                    "at least three counterfactual answers are required"
                )
