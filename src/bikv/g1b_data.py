from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random

from bikv.data import AGENTS, LABS, _code, _family_entities, _record_context
from bikv.schema import StorySample


def qualification_plan(sample: StorySample, budget: int, seed: int) -> dict[str, object]:
    """Build full/evidence-oracle masks without learned feature capture."""
    required = sorted(set(sample.supporting_blocks))
    if not required or len(required) > budget:
        raise ValueError("Qualification evidence does not fit the requested budget")
    sample_seed = seed + int(hashlib.sha1(sample.sample_id.encode()).hexdigest()[:8], 16)
    available = [index for index in range(len(sample.blocks)) if index not in required]
    fillers = random.Random(sample_seed ^ 0x51414C).sample(available, budget - len(required))
    return {
        "masks": {"full": None, "task_oracle": sorted(required + fillers)},
        "candidate_pool": [],
        "policy_scores": {},
    }


def select_g1b_family_subset(samples: list[StorySample], story_limit: int) -> list[StorySample]:
    """Select complete counterfactual families while preserving condition/position coverage."""
    if story_limit <= 0 or story_limit > len(samples) or story_limit % 2:
        raise ValueError("g1b_eval_stories_per_length must be a positive even number within the dataset")
    if story_limit == len(samples):
        return samples

    families: dict[str, list[StorySample]] = defaultdict(list)
    for sample in samples:
        families[str(sample.metadata["counterfactual_group"])].append(sample)
    if any(len(rows) != 2 for rows in families.values()):
        raise ValueError("G1b evaluation subset requires complete two-story counterfactual families")

    by_condition: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for family, rows in families.items():
        conditions = {str(row.metadata["evidence_condition"]) for row in rows}
        if len(conditions) != 1:
            raise ValueError(f"G1b family {family} changes evidence condition")
        position = min(min(row.supporting_blocks) for row in rows)
        by_condition[conditions.pop()].append((position, family))

    target_families = story_limit // 2
    total_families = len(families)
    quotas = {
        condition: math.floor(target_families * len(rows) / total_families)
        for condition, rows in by_condition.items()
    }
    remaining = target_families - sum(quotas.values())
    remainders = sorted(
        by_condition,
        key=lambda condition: (
            target_families * len(by_condition[condition]) / total_families - quotas[condition],
            condition,
        ),
        reverse=True,
    )
    for condition in remainders[:remaining]:
        quotas[condition] += 1

    selected_families: set[str] = set()
    for condition, rows in sorted(by_condition.items()):
        ordered = sorted(rows)
        count = quotas[condition]
        if count == 0:
            continue
        if count == 1:
            indices = [len(ordered) // 2]
        else:
            indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
        selected_families.update(ordered[index][1] for index in indices)

    selected = [sample for sample in samples if str(sample.metadata["counterfactual_group"]) in selected_families]
    if len(selected) != story_limit:
        raise RuntimeError(f"Expected {story_limit} selected G1b stories, got {len(selected)}")
    return selected


CODE_TEMPLATES = (
    "Credential register. The verification code assigned to technician {agent} is {code}.",
    "Access ledger. Technician {agent} uses verification credential {code}.",
    "Security record. The active verification code for technician {agent} is {code}.",
)
ROSTER_TEMPLATES = (
    "Laboratory roster. The technician assigned to laboratory {lab} is {agent}.",
    "Personnel assignment. Laboratory {lab} is operated by technician {agent}.",
    "Shift register. Technician {agent} is responsible for laboratory {lab}.",
)
QUESTIONS = (
    "Identify the technician assigned to laboratory {lab}, then return that technician's verification code.",
    "Which verification code belongs to the technician responsible for laboratory {lab}?",
    "Follow the laboratory-to-technician and technician-to-code records for {lab}. Return only the code.",
)
AUDIT_TEMPLATES = (
    "Audit note. Technician {agent} reviewed laboratory {lab}; reference {code} was unrelated to access.",
    "Maintenance log. Laboratory {lab} was inspected by {agent}; ticket {code} is not a credential.",
    "Archive cross-check. {agent} signed a report for {lab}; document number {code} does not grant access.",
)


def _balanced_positions(n_blocks: int, family_count: int, seed: int) -> list[int]:
    if family_count == 1:
        return [(n_blocks - 2) // 2]
    positions = [round(index * (n_blocks - 2) / (family_count - 1)) for index in range(family_count)]
    random.Random(seed ^ (n_blocks * 104729)).shuffle(positions)
    return positions


def _separated_pair(anchor: int, n_blocks: int) -> tuple[int, int]:
    # Anchors enumerate boundaries (0..n-2). Preserve coverage of the final
    # document block when the final boundary belongs to the separated control.
    if anchor == n_blocks - 2:
        return tuple(sorted((n_blocks - 1, n_blocks // 2)))
    other = (anchor + max(3, n_blocks // 2)) % n_blocks
    if abs(other - anchor) <= 1:
        other = (other + 2) % n_blocks
    return tuple(sorted((anchor, other)))


def _condition_schedule(family_count: int, seed: int) -> list[str]:
    separated = max(1, round(family_count / 6))
    schedule = ["adjacent"] * (family_count - separated) + ["separated"] * separated
    random.Random(seed ^ 0x71B0C0DE).shuffle(schedule)
    return schedule


def _story(
    story_index: int,
    n_stories: int,
    n_blocks: int,
    seed: int,
    anchor: int,
    condition: str,
) -> StorySample:
    family_index = story_index // 2
    counterfactual_index = story_index % 2
    agents, labs, codes = _family_entities(family_index, seed)
    assignment_shift = story_index % len(agents)
    lab_to_agent = [agents[(index + assignment_shift) % len(agents)] for index in range(3)]
    target_lab = labs[0]
    target_agent = lab_to_agent[0]
    target_agent_index = agents.index(target_agent)
    answer = codes[target_agent_index]
    # Counterfactual family members share wording, distractors and placement.
    # Only the laboratory assignment, and therefore the answer, changes.
    rng = random.Random(seed + family_index * 1009)

    code_records = [
        CODE_TEMPLATES[(family_index + index) % len(CODE_TEMPLATES)].format(agent=agent, code=code)
        + _record_context(family_index, index)
        for index, (agent, code) in enumerate(zip(agents, codes))
    ]
    roster_records = [
        ROSTER_TEMPLATES[(family_index + index) % len(ROSTER_TEMPLATES)].format(lab=lab, agent=agent)
        + _record_context(family_index, index + len(code_records))
        for index, (lab, agent) in enumerate(zip(labs, lab_to_agent))
    ]
    target_code = code_records[target_agent_index]
    target_roster = roster_records[0]
    distractors = [row for row in code_records + roster_records if row not in {target_code, target_roster}]
    used_codes = set(codes)
    while len(distractors) < n_blocks - 2:
        audit_code = _code(rng)
        while audit_code in used_codes:
            audit_code = _code(rng)
        used_codes.add(audit_code)
        distractors.append(
            rng.choice(AUDIT_TEMPLATES).format(
                agent=rng.choice(agents), lab=rng.choice(labs[1:]), code=audit_code
            )
            + _record_context(family_index, len(distractors))
        )
    rng.shuffle(distractors)

    if condition == "adjacent":
        support = [anchor, anchor + 1]
    elif condition == "separated":
        support = list(_separated_pair(anchor, n_blocks))
    else:
        raise ValueError(f"Unsupported G1b condition: {condition}")
    evidence_order = "code_then_roster" if family_index % 2 == 0 else "roster_then_code"
    evidence = [target_code, target_roster] if evidence_order == "code_then_roster" else [target_roster, target_code]
    blocks: list[str | None] = [None] * n_blocks
    for position, row in zip(support, evidence):
        blocks[position] = row
    distractor_iter = iter(distractors[: n_blocks - 2])
    for index in range(n_blocks):
        if blocks[index] is None:
            blocks[index] = next(distractor_iter)

    code_by_agent = dict(zip(agents, codes))
    possible_without_roster = [agent for agent in agents if agent not in set(lab_to_agent[1:])]
    metadata = {
        "generator_version": 8,
        "generator_protocol": "g1b_stratified_independent",
        "counterfactual_group": f"g1b-{n_blocks:03d}-{family_index:04d}",
        "counterfactual_index": counterfactual_index,
        "candidate_agents": agents,
        "candidate_labs": labs,
        "candidate_codes": codes,
        "target_agent": target_agent,
        "target_lab": target_lab,
        "audit_target_lab_mentions": 0,
        "support_boundary": support[0] if condition == "adjacent" else None,
        "evidence_order": evidence_order,
        "evidence_condition": condition,
        "code_block": support[0] if evidence_order == "code_then_roster" else support[1],
        "roster_block": support[1] if evidence_order == "code_then_roster" else support[0],
        "symbolic_ablation_candidates": {
            "without_left": ["UNOBSERVED_A", "UNOBSERVED_B", "UNOBSERVED_C"],
            "without_right": [code_by_agent[agent] for agent in possible_without_roster],
            "without_both": ["UNOBSERVED_A", "UNOBSERVED_B", "UNOBSERVED_C"],
        },
        "generator_seed": seed + family_index * 1009,
    }
    return StorySample(
        sample_id=f"g1b-{n_blocks:03d}-{story_index:04d}",
        story_id=f"g1b-{n_blocks:03d}-{story_index:04d}",
        split="test",
        variant="neutral",
        blocks=[str(block) for block in blocks],
        question=QUESTIONS[family_index % len(QUESTIONS)].format(lab=target_lab),
        answer=answer,
        answer_aliases=[answer, answer.lower(), answer.replace("-", "")],
        supporting_blocks=support,
        manipulated_boundary=min(support[0], n_blocks - 2),
        boundary_family=condition,
        metadata=metadata,
    )


def generate_g1b_dataset(n_stories: int, n_blocks: int, seed: int) -> list[StorySample]:
    if n_stories < 2 or n_stories % 2:
        raise ValueError("G1b requires an even number of stories to form counterfactual pairs")
    if n_blocks < 10:
        raise ValueError("G1b requires at least ten blocks")
    family_count = math.ceil(n_stories / 2)
    positions = _balanced_positions(n_blocks, family_count, seed)
    conditions = _condition_schedule(family_count, seed)
    samples = []
    for story_index in range(n_stories):
        family_index = story_index // 2
        samples.append(
            _story(
                story_index,
                n_stories,
                n_blocks,
                seed,
                positions[family_index],
                conditions[family_index],
            )
        )
    return samples
