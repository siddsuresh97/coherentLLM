"""Build the NOVA target similarity matrix and held-out concept split.

This is step 1 of the coherence-SFT pipeline:
  * S* is cosine similarity over NOVA binary feature rows.
  * test is the fixed 128 THINGS concepts already used by the eval pipeline.
  * train is every other NOVA concept after deterministic near-leak exclusion.
"""
import difflib
import os
import re

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOVA_PATH = os.path.join(ROOT, "data", "nova", "verified_matrix_cogsci2025.parquet")
TEST_PATH = os.path.join(ROOT, "data", "scale128", "concepts.csv")
OUT_DIR = os.path.join(ROOT, "data", "sft")
S_STAR_PATH = os.path.join(OUT_DIR, "S_star.npy")
S_STAR_CONCEPTS_PATH = os.path.join(OUT_DIR, "S_star_concepts.csv")
TRAIN_PATH = os.path.join(OUT_DIR, "train_concepts.csv")
TEST_OUT_PATH = os.path.join(OUT_DIR, "test_concepts.csv")


def _clean_concept(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())


def _tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", _clean_concept(s)) if t]


def _compact(s):
    return "".join(_tokens(s))


def _singular_token(t):
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("ves"):
        return t[:-3] + "f"
    if len(t) > 3 and t.endswith("es") and not t.endswith(("ses", "xes")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("us", "is", "ss")):
        return t[:-1]
    return t


def _canonical(s):
    return "".join(_singular_token(t) for t in _tokens(s))


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _leak_reason(candidate, test_concept):
    """Return a conservative string-leak reason, or None.

    WordNet/NLTK is not installed in the requested env, so this intentionally
    stays deterministic and auditable: morphology, word-boundary compounds,
    obvious typos, and compact aliases such as plane/airplane.
    """
    cand_tokens = _tokens(candidate)
    test_tokens = _tokens(test_concept)
    cand_compact = _compact(candidate)
    test_compact = _compact(test_concept)

    if _canonical(candidate) == _canonical(test_concept):
        return "morphological variant"

    cand_set = {_singular_token(t) for t in cand_tokens}
    test_set = {_singular_token(t) for t in test_tokens}
    if len(cand_tokens) > 1 and test_set <= cand_set:
        return "token-boundary hypernym/string match"
    if len(test_tokens) > 1 and cand_set <= test_set:
        return "token-boundary hypernym/string match"

    ratio = difflib.SequenceMatcher(None, cand_compact, test_compact).ratio()
    edit_distance = _edit_distance(cand_compact, test_compact)
    if (
        min(len(cand_compact), len(test_compact)) >= 5
        and edit_distance <= 2
        and ratio >= 0.88
        and cand_compact[:3] == test_compact[:3]
    ):
        return f"typo/near string match ({ratio:.2f})"

    short, long = (
        (cand_compact, test_compact)
        if len(cand_compact) <= len(test_compact)
        else (test_compact, cand_compact)
    )
    if min(len(short), len(long)) >= 4 and long.endswith(short):
        if len(long) - len(short) >= 3 and ratio >= 0.75:
            return f"compact alias/string containment ({ratio:.2f})"

    return None


def _load_concept_lines(path):
    with open(path) as f:
        return [_clean_concept(ln) for ln in f if ln.strip()]


def _write_concepts(path, concepts):
    pd.Series(concepts).to_csv(path, index=False, header=False)


def _load_nova():
    df = pd.read_parquet(NOVA_PATH)
    concepts = [_clean_concept(c) for c in df.index]
    if len(concepts) != len(set(concepts)):
        dupes = sorted({c for c in concepts if concepts.count(c) > 1})
        raise ValueError(f"NOVA concepts are not unique after lowercase/strip: {dupes[:20]}")
    df.index = concepts

    values = df.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("NOVA matrix contains non-finite values")
    if not np.logical_or(values == 0.0, values == 1.0).all():
        raise ValueError("NOVA matrix must be binary 0/1")
    if (values.sum(axis=1) == 0).any():
        zero_concepts = df.index[values.sum(axis=1) == 0].tolist()
        raise ValueError(f"NOVA has zero-feature concepts: {zero_concepts[:20]}")
    return df.index.tolist(), values


def _cosine_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / norms
    sim = normalized @ normalized.T
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32, copy=False)


def _find_leakage(candidates, test_concepts):
    leaks = []
    for candidate in candidates:
        for test_concept in test_concepts:
            reason = _leak_reason(candidate, test_concept)
            if reason is not None:
                leaks.append((candidate, test_concept, reason))
                break
    return leaks


def _build_split(nova_concepts, eval_concepts):
    nova_set = set(nova_concepts)
    test_in_nova = [c for c in eval_concepts if c in nova_set]
    missing = [c for c in eval_concepts if c not in nova_set]
    test_set = set(test_in_nova)

    train_pool = [c for c in nova_concepts if c not in test_set]
    leaks = _find_leakage(train_pool, test_in_nova)
    leak_set = {candidate for candidate, _, _ in leaks}
    train = [c for c in train_pool if c not in leak_set]

    assert set(train).isdisjoint(test_set)
    remaining_leaks = _find_leakage(train, test_in_nova)
    assert not remaining_leaks, remaining_leaks[:10]
    return train, test_in_nova, missing, leaks


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    nova_concepts, values = _load_nova()
    eval_concepts = _load_concept_lines(TEST_PATH)
    if len(eval_concepts) != len(set(eval_concepts)):
        raise ValueError("Eval concept list contains duplicates after lowercase/strip")

    s_star = _cosine_rows(values)
    train, test, missing, leaks = _build_split(nova_concepts, eval_concepts)

    np.save(S_STAR_PATH, s_star)
    _write_concepts(S_STAR_CONCEPTS_PATH, nova_concepts)
    _write_concepts(TRAIN_PATH, train)
    _write_concepts(TEST_OUT_PATH, test)

    print("SFT target build complete")
    print(f"|NOVA|: {len(nova_concepts)}")
    print(f"S*: {s_star.shape[0]} x {s_star.shape[1]}")
    print(f"|eval concepts|: {len(eval_concepts)}")
    print(f"|test in NOVA|: {len(test)}")
    print(f"|train|: {len(train)}")
    print(f"#excluded-as-leakage: {len(leaks)}")
    print(f"missing eval concepts from NOVA: {missing if missing else 'none'}")
    print("excluded leakage candidates:")
    if leaks:
        for candidate, matched, reason in leaks:
            print(f"  {candidate} -> {matched} ({reason})")
    else:
        print("  none")
    print(f"sample train: {train[:10]}")
    print(f"sample test: {test[:10]}")
    print("sanity: train/test exact overlap = 0; heuristic leakage after exclusions = 0")
    print(f"wrote: {os.path.relpath(S_STAR_PATH, ROOT)}")
    print(f"wrote: {os.path.relpath(S_STAR_CONCEPTS_PATH, ROOT)}")
    print(f"wrote: {os.path.relpath(TRAIN_PATH, ROOT)}")
    print(f"wrote: {os.path.relpath(TEST_OUT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
