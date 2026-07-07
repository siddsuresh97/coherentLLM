"""Load stimuli and build the (row -> prompt) job list for each method."""
import csv
import os
from typing import List, Tuple

from prompts import BUILDERS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STIM = os.environ.get("COHERENCE_STIM_DIR", os.path.join(HERE, "data", "stimuli"))


def _read_rows(path: str) -> List[List[str]]:
    with open(path) as f:
        return [[c.strip() for c in row] for row in csv.reader(f) if any(row)]


def load_concepts() -> List[str]:
    with open(os.path.join(STIM, "concepts.csv")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_jobs(method: str, feature_sample: int = 0,
               feature_file: str = "features.csv") -> List[Tuple[Tuple[str, ...], str]]:
    """Return a list of ((input tuple), prompt string) for a method.

    triplet  -> rows of (anchor, c1, c2) from triplets.csv
    pairwise -> rows of (a, b) from pairs.csv
    feature  -> cartesian product of features x concepts. `feature_file` selects
                which feature list (e.g. features_discriminative.csv); feature_sample
                caps the number of features used (0 = all).
    """
    builder, arity = BUILDERS[method]
    jobs = []
    if method == "triplet":
        for row in _read_rows(os.path.join(STIM, "triplets.csv")):
            a, c1, c2 = row[:3]
            jobs.append(((a, c1, c2), builder(a, c1, c2)))
    elif method == "pairwise":
        for row in _read_rows(os.path.join(STIM, "pairs.csv")):
            a, b = row[:2]
            jobs.append(((a, b), builder(a, b)))
    elif method == "feature":
        concepts = load_concepts()
        feats = [r[0] for r in _read_rows(os.path.join(STIM, feature_file))]
        if feature_sample and feature_sample < len(feats):
            feats = feats[:feature_sample]
        for feat in feats:
            for concept in concepts:
                jobs.append(((feat, concept), builder(feat, concept)))
    else:
        raise ValueError(f"unknown method {method}")
    return jobs
