"""Prompt paraphrases for SFT coherence consistency checks.

Template 0 for each method is the canonical wording from src/prompts.py.
The remaining templates keep the same answer space while changing surface form.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from prompts import feature_prompt, pairwise_prompt, triplet_prompt  # noqa: E402


TripletTemplate = Callable[[str, str, str], str]
PairwiseTemplate = Callable[[str, str], str]
FeatureTemplate = Callable[[str, str], str]


TRIPLET_TEMPLATES: List[TripletTemplate] = [
    triplet_prompt,
    lambda anchor, c1, c2: (
        f"Choose only {c1} or {c2}. Which one is closer in meaning to {anchor}?"
    ),
    lambda anchor, c1, c2: (
        f"Relative to {anchor}, which item is semantically more similar: {c1} or {c2}? "
        "Reply with just the item name."
    ),
    lambda anchor, c1, c2: (
        f"Pick the better semantic match for {anchor} from these two options: {c1}; {c2}. "
        "Answer with one option only."
    ),
    lambda anchor, c1, c2: (
        f"Between {c1} and {c2}, which concept has a meaning more like {anchor}? "
        "Use only the chosen concept."
    ),
]


PAIRWISE_TEMPLATES: List[PairwiseTemplate] = [
    pairwise_prompt,
    lambda a, b: (
        f"Rate the semantic similarity of {a} and {b} from 1 to 7, where 1 means "
        "extremely dissimilar and 7 means extremely similar. Answer with only the number."
    ),
    lambda a, b: (
        f"Using a 1-7 scale for meaning similarity, how close are {a} and {b}? "
        "1 is least similar, 7 is most similar. Give one number."
    ),
    lambda a, b: (
        f"How semantically alike are {a} and {b}? Respond with a single integer from "
        "1 (not alike) to 7 (very alike)."
    ),
    lambda a, b: (
        f"Assign a similarity score for {a} versus {b}: 1 = extremely different, "
        "7 = extremely similar. Return only a number from 1 to 7."
    ),
]


FEATURE_TEMPLATES: List[FeatureTemplate] = [
    feature_prompt,
    lambda feature, concept: (
        f"Is the property [{feature}] true for the concept [{concept}]? "
        "Answer only True or False."
    ),
    lambda feature, concept: (
        f"For [{concept}], does the property [{feature}] apply? Reply with exactly "
        "True or False."
    ),
    lambda feature, concept: (
        f"Judge this concept-property statement: [{concept}] has property [{feature}]. "
        "Use one word: True or False."
    ),
    lambda feature, concept: (
        f"Can [{feature}] be considered a true property of [{concept}]? "
        "Respond True or False only."
    ),
]


TEMPLATES: Dict[str, List[Callable]] = {
    "triplet": TRIPLET_TEMPLATES,
    "pairwise": PAIRWISE_TEMPLATES,
    "feature": FEATURE_TEMPLATES,
}

