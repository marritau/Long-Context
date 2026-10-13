from __future__ import annotations

from collections import Counter
import math
import random
import re
import string

from bikv.schema import StorySample


CHAIN_LENGTHS = (2, 3, 4)
PLACEMENTS = ("adjacent", "dispersed")


def _token(rng: random.Random, prefix: str) -> str:
    suffix = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
    return f"{prefix}-{suffix}"


def _code(rng: random.Random) -> str:
    return f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randrange(100, 1000)}"


def _support_positions(
    n_blocks: int, chain_length: int, placement: str, family_index: int
) -> list[int]:
    if placement == "adjacent":
        start = (family_index * 7) % (n_blocks - chain_length + 1)
        return list(range(start, start + chain_length))
    if placement != "dispersed":
        raise ValueError(f"Unsupported placement: {placement}")
    # Evenly spread the chain while rotating positions between families.
    anchors = [round(index * (n_blocks - 1) / (chain_length - 1)) for index in range(chain_length)]
    shift = family_index % max(1, n_blocks // 7)
    positions = sorted({(value + shift) % n_blocks for value in anchors})
    if len(positions) != chain_length or min(
        right - left for left, right in zip(positions, positions[1:])
    ) < 2:
        positions = [round((index + 0.5) * n_blocks / chain_length) for index in range(chain_length)]
        positions = [min(n_blocks - 1, value) for value in positions]
    if len(set(positions)) != chain_length:
        raise RuntimeError("Could not construct distinct dispersed support positions")
    return sorted(positions)


def _records_for_family(
    family_index: int, counterfactual_index: int, chain_length: int, seed: int
) -> tuple[list[list[str]], list[str], list[str], str, str]:
    rng = random.Random(seed + family_index * 10007)
    labs = [_token(rng, "LAB") for _ in range(6)]
    agents = [_token(rng, "TECH") for _ in range(6)]
    vaults = [_token(rng, "VAULT") for _ in range(6)]
    terminals = [_token(rng, "TERM") for _ in range(6)]
    codes: list[str] = []
    while len(codes) < 6:
        value = _code(rng)
        if value not in codes:
            codes.append(value)

    # Paired stories swap only two laboratory assignments. Downstream chains
    # stay fixed, so the answer changes without changing the relation schema.
    assignment = list(range(6))
    if counterfactual_index:
        assignment[0], assignment[1] = assignment[1], assignment[0]

    chains: list[list[str]] = []
    for lab_index, lab in enumerate(labs):
        chain_index = assignment[lab_index]
        agent = agents[chain_index]
        vault = vaults[chain_index]
        terminal = terminals[chain_index]
        if chain_length == 2:
            records = [
                f"Assignment record: laboratory {lab} is operated by technician {agent}.",
                f"Credential record: technician {agent} uses verification code {codes[chain_index]}.",
            ]
        elif chain_length == 3:
            records = [
                f"Assignment record: laboratory {lab} is operated by technician {agent}.",
                f"Storage record: technician {agent} is assigned vault {vault}.",
                f"Credential record: vault {vault} uses verification code {codes[chain_index]}.",
            ]
        elif chain_length == 4:
            records = [
                f"Assignment record: laboratory {lab} is operated by technician {agent}.",
                f"Storage record: technician {agent} is assigned vault {vault}.",
                f"Routing record: vault {vault} forwards verification through terminal {terminal}.",
                f"Credential record: terminal {terminal} uses verification code {codes[chain_index]}.",
            ]
        else:
            raise ValueError(f"Unsupported chain length: {chain_length}")
        chains.append(records)
    return chains, labs, codes, labs[0], codes[assignment[0]]


def _story(
    story_index: int,
    n_stories: int,
    n_blocks: int,
    seed: int,
    chain_length: int,
    placement: str,
) -> StorySample:
    family_index = story_index // 2
    counterfactual_index = story_index % 2
    chains, labs, codes, target_lab, answer = _records_for_family(
        family_index, counterfactual_index, chain_length, seed
    )
    supports = _support_positions(n_blocks, chain_length, placement, family_index)
    target_records = chains[0]
    distractors = [record for chain in chains[1:] for record in chain]
    rng = random.Random(seed + family_index * 15401)
    while len(distractors) < n_blocks - chain_length:
        ticket = _token(rng, "DOC")
        distractors.append(
            f"Administrative note: archive ticket {ticket} records a completed inventory check; "
            "it does not define any assignment, route, or verification code."
        )
    rng.shuffle(distractors)

    blocks: list[str | None] = [None] * n_blocks
    for position, record in zip(supports, target_records):
        blocks[position] = record
    distractor_iter = iter(distractors)
    for index in range(n_blocks):
        if blocks[index] is None:
            blocks[index] = next(distractor_iter)

    family = f"h-chain-{chain_length}-{placement}-{family_index:04d}"
    return StorySample(
        sample_id=f"{family}-{counterfactual_index}",
        story_id=f"{family}-{counterfactual_index}",
        split="test",
        variant="neutral",
        blocks=[str(value) for value in blocks],
        question=(
            f"Follow every assignment, storage, and routing record beginning at laboratory {target_lab}. "
            "Return only the final verification code."
        ),
        answer=answer,
        answer_aliases=[answer, answer.lower(), answer.replace("-", "")],
        supporting_blocks=supports,
        manipulated_boundary=min(supports[0], n_blocks - 2),
        boundary_family=f"chain_{placement}",
        metadata={
            "generator_version": 1,
            "generator_protocol": "stage_h_distance_balanced_chains",
            "counterfactual_group": family,
            "counterfactual_index": counterfactual_index,
            "candidate_codes": codes,
            "candidate_labs": labs,
            "target_lab": target_lab,
            "chain_length": chain_length,
            "evidence_condition": placement,
            "support_span": supports[-1] - supports[0],
            "support_gaps": [right - left for left, right in zip(supports, supports[1:])],
            "generator_seed": seed + family_index * 15401,
        },
    )


def generate_chain_dataset(n_stories: int, n_blocks: int, seed: int) -> list[StorySample]:
    strata = [(length, placement) for length in CHAIN_LENGTHS for placement in PLACEMENTS]
    if n_stories < len(strata) * 2 or n_stories % (len(strata) * 2):
        raise ValueError("Stage H story count must be a positive multiple of 12")
    if n_blocks < 24:
        raise ValueError("Stage H requires at least 24 blocks for complete distractor chains")
    family_count = n_stories // 2
    samples = []
    for family_index in range(family_count):
        chain_length, placement = strata[family_index % len(strata)]
        for counterfactual_index in range(2):
            story_index = family_index * 2 + counterfactual_index
            samples.append(
                _story(
                    story_index,
                    n_stories,
                    n_blocks,
                    seed,
                    chain_length,
                    placement,
                )
            )
    audit_chain_dataset(samples)
    return samples


def audit_chain_dataset(samples: list[StorySample]) -> dict[str, object]:
    families: dict[str, list[StorySample]] = {}
    for sample in samples:
        sample.validate()
        families.setdefault(str(sample.metadata["counterfactual_group"]), []).append(sample)
    if any(len(rows) != 2 for rows in families.values()):
        raise ValueError("Every Stage H counterfactual family must contain two stories")
    for family, rows in families.items():
        if len({row.answer for row in rows}) != 2:
            raise ValueError(f"Counterfactual family {family} does not change the answer")
        if rows[0].supporting_blocks != rows[1].supporting_blocks:
            raise ValueError(f"Counterfactual family {family} changes evidence positions")
        for row in rows:
            target = str(row.metadata["target_lab"])
            mentions = sum(bool(re.search(rf"\b{re.escape(target)}\b", block)) for block in row.blocks)
            if mentions != 1:
                raise ValueError(f"{row.sample_id} contains {mentions} target-lab mentions")
    strata = Counter(
        (int(row.metadata["chain_length"]), str(row.metadata["evidence_condition"]))
        for rows in families.values()
        for row in rows[:1]
    )
    if max(strata.values()) - min(strata.values()) > 1:
        raise ValueError("Stage H chain/distance strata are not balanced")
    return {
        "stories": len(samples),
        "families": len(families),
        "strata": {f"h{length}_{placement}": count for (length, placement), count in sorted(strata.items())},
    }
