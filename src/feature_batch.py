"""Batched feature-verification: ask one prompt to mark True/False for a block of
features for a single concept. Cuts API calls ~20x vs one-pair-per-call.

Prompt asks for a compact numbered T/F list; the parser recovers per-feature answers.
Output CSV matches the single-pair schema (input=feature|concept, response=True/False)
so downstream analysis (analysis.feature_similarity) is unchanged.
"""
import re
from typing import List, Tuple

BATCH_INSTRUCTIONS = (
    "For the concept [{concept}], say whether each property below is true or false.\n"
    "Answer with the number and then True or False only, one per line, e.g. `1. True`.\n"
    "Properties:\n{items}"
)


def make_batches(features: List[str], concepts: List[str], block: int = 20):
    """Yield (concept, [features], prompt) for each block of features x concept."""
    for concept in concepts:
        for i in range(0, len(features), block):
            chunk = features[i:i + block]
            items = "\n".join(f"{j+1}. {f}" for j, f in enumerate(chunk))
            prompt = BATCH_INSTRUCTIONS.format(concept=concept, items=items)
            yield concept, chunk, prompt


def parse_batch(response: str, chunk: List[str]) -> List[Tuple[str, int]]:
    """Return [(feature, 0/1)] parsed from a numbered T/F response.

    Robust to missing/extra lines: matches by leading number; unmatched features
    are dropped (not guessed)."""
    ans = {}
    for line in str(response).splitlines():
        m = re.match(r"\s*(\d+)\s*[\.\):]?\s*(.*)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        rest = m.group(2).lower()
        if 0 <= idx < len(chunk):
            if "true" in rest:
                ans[idx] = 1
            elif "false" in rest:
                ans[idx] = 0
    return [(chunk[i], ans[i]) for i in range(len(chunk)) if i in ans]
