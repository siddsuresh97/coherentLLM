"""Generate mutually-consistent SFT data from the NOVA-derived S* target.

The real arm labels triplet/pairwise/feature prompts from the same train-only
NOVA geometry. The scrambled control keeps the exact same prompts but labels
them through one fixed permutation of the train concept index.
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from prompts import feature_prompt, pairwise_prompt, triplet_prompt  # noqa: E402


NOVA_PATH = os.path.join(ROOT, "data", "nova", "verified_matrix_cogsci2025.parquet")
S_STAR_PATH = os.path.join(ROOT, "data", "sft", "S_star.npy")
S_STAR_CONCEPTS_PATH = os.path.join(ROOT, "data", "sft", "S_star_concepts.csv")
TRAIN_CONCEPTS_PATH = os.path.join(ROOT, "data", "sft", "train_concepts.csv")
TEST_CONCEPTS_PATH = os.path.join(ROOT, "data", "sft", "test_concepts.csv")
OUT_PATH = os.path.join(ROOT, "data", "sft", "train.jsonl")
SCRAMBLED_OUT_PATH = os.path.join(ROOT, "data", "sft", "train_scrambled.jsonl")

SEED = 0
N_TRIPLET = 60_000
N_PAIRWISE = 30_000
N_FEATURE = 20_000
N_CLUSTERS = 15


def _clean_text(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())


def _load_concept_lines(path):
    with open(path) as f:
        return [_clean_text(ln) for ln in f if ln.strip()]


def _chat_example(prompt, response):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(response)},
        ]
    }


def _concept_patterns(concepts):
    return [
        (concept, re.compile(r"(?<![a-z0-9])" + re.escape(concept) + r"(?![a-z0-9])"))
        for concept in concepts
    ]


def _load_inputs():
    s_star = np.load(S_STAR_PATH)
    s_concepts = _load_concept_lines(S_STAR_CONCEPTS_PATH)
    train = _load_concept_lines(TRAIN_CONCEPTS_PATH)
    test = _load_concept_lines(TEST_CONCEPTS_PATH)

    if s_star.shape != (len(s_concepts), len(s_concepts)):
        raise ValueError(
            f"S* shape {s_star.shape} does not match {len(s_concepts)} concepts"
        )
    if len(s_concepts) != len(set(s_concepts)):
        raise ValueError("S* concept list contains duplicates")
    if len(train) != len(set(train)):
        raise ValueError("Train concept list contains duplicates")
    if len(test) != len(set(test)):
        raise ValueError("Test concept list contains duplicates")
    if set(train) & set(test):
        raise ValueError(f"Train/test overlap: {sorted(set(train) & set(test))[:20]}")

    s_index = {concept: i for i, concept in enumerate(s_concepts)}
    missing_train = [concept for concept in train if concept not in s_index]
    missing_test = [concept for concept in test if concept not in s_index]
    if missing_train or missing_test:
        raise ValueError(
            f"Split concepts missing from S*: train={missing_train[:10]}, "
            f"test={missing_test[:10]}"
        )

    nova = pd.read_parquet(NOVA_PATH)
    nova.index = [_clean_text(c) for c in nova.index]
    if len(nova.index) != len(set(nova.index)):
        raise ValueError("NOVA concepts are not unique after lowercase/strip")
    nova = nova.reindex(s_concepts)
    if nova.isna().any().any():
        raise ValueError("NOVA matrix could not be aligned to S* concepts")

    feature_names = [_clean_text(c) for c in nova.columns]
    if len(feature_names) != len(set(feature_names)):
        dupes = [k for k, v in Counter(feature_names).items() if v > 1]
        raise ValueError(f"Duplicate normalized feature names: {dupes[:20]}")

    test_patterns = _concept_patterns(test)
    keep_features = np.array(
        [
            not any(pattern.search(feature) for _, pattern in test_patterns)
            for feature in feature_names
        ],
        dtype=bool,
    )
    excluded_features = [
        feature for feature, keep in zip(feature_names, keep_features) if not keep
    ]
    feature_names = [
        feature for feature, keep in zip(feature_names, keep_features) if keep
    ]

    values = nova.to_numpy(dtype=np.int8, copy=True)[:, keep_features]
    if not np.logical_or(values == 0, values == 1).all():
        raise ValueError("NOVA matrix must be binary 0/1")

    train_idx = np.array([s_index[c] for c in train], dtype=np.int64)
    s_train = s_star[np.ix_(train_idx, train_idx)]
    m_train = values[train_idx]
    return s_star, s_train, m_train, feature_names, excluded_features, train, test


def _label_triplet(sim, anchor_i, b_i, c_i):
    b_sim = sim[anchor_i, b_i]
    c_sim = sim[anchor_i, c_i]
    if np.isclose(b_sim, c_sim, rtol=0.0, atol=1e-8):
        return None
    return b_i if b_sim > c_sim else c_i


def _sample_other_from_cluster(rng, cluster_members, cluster_id, excluded):
    members = [i for i in cluster_members[cluster_id] if i not in excluded]
    if not members:
        return None
    return int(rng.choice(members))


def _sample_other_outside_cluster(rng, cluster_members, cluster_id, excluded):
    choices = [
        i
        for cid, members in cluster_members.items()
        if cid != cluster_id
        for i in members
        if i not in excluded
    ]
    if not choices:
        return None
    return int(rng.choice(choices))


def _generate_triplets(rng, s_train, s_scrambled, concepts):
    n = len(concepts)
    labels = KMeans(
        n_clusters=min(N_CLUSTERS, n),
        random_state=SEED,
        n_init=20,
    ).fit_predict(s_train)
    cluster_members = {
        cluster_id: np.flatnonzero(labels == cluster_id).astype(int).tolist()
        for cluster_id in sorted(set(labels))
    }

    real = []
    scrambled = []
    metadata = []
    attempts = 0
    while len(real) < N_TRIPLET:
        attempts += 1
        if attempts > N_TRIPLET * 200:
            raise RuntimeError("Could not sample enough non-tied triplets")

        anchor_i = int(rng.integers(n))
        anchor_cluster = int(labels[anchor_i])
        mode = "within" if rng.random() < 0.5 else "across"

        if mode == "within" and len(cluster_members[anchor_cluster]) >= 3:
            b_i = _sample_other_from_cluster(rng, cluster_members, anchor_cluster, {anchor_i})
            c_i = _sample_other_from_cluster(
                rng, cluster_members, anchor_cluster, {anchor_i, b_i}
            )
        else:
            b_i = _sample_other_from_cluster(rng, cluster_members, anchor_cluster, {anchor_i})
            c_i = _sample_other_outside_cluster(
                rng, cluster_members, anchor_cluster, {anchor_i, b_i}
            )
            mode = "across"

        if b_i is None or c_i is None or b_i == c_i:
            continue
        if rng.random() < 0.5:
            b_i, c_i = c_i, b_i

        real_label_i = _label_triplet(s_train, anchor_i, b_i, c_i)
        scrambled_label_i = _label_triplet(s_scrambled, anchor_i, b_i, c_i)
        if real_label_i is None or scrambled_label_i is None:
            continue

        anchor = concepts[anchor_i]
        b = concepts[b_i]
        c = concepts[c_i]
        prompt = triplet_prompt(anchor, b, c)
        real.append(_chat_example(prompt, concepts[real_label_i]))
        scrambled.append(_chat_example(prompt, concepts[scrambled_label_i]))
        metadata.append(
            {
                "view": "triplet",
                "mode": mode,
                "concepts": (anchor, b, c),
                "real_label": concepts[real_label_i],
                "scrambled_label": concepts[scrambled_label_i],
            }
        )

    return real, scrambled, metadata


def _rating_mapper(s_train):
    tri = s_train[np.triu_indices_from(s_train, k=1)]
    lo = float(tri.min())
    hi = float(tri.max())
    if not hi > lo:
        raise ValueError("Train S* off-diagonal similarities are degenerate")

    def to_rating(sim):
        scaled = 1.0 + 6.0 * (float(sim) - lo) / (hi - lo)
        return int(np.clip(np.floor(scaled + 0.5), 1, 7))

    return to_rating, lo, hi


def _generate_pairwise(rng, s_train, s_scrambled, concepts):
    n = len(concepts)
    all_i, all_j = np.triu_indices(n, k=1)
    if N_PAIRWISE > len(all_i):
        raise ValueError(f"Requested {N_PAIRWISE} pairs, but only {len(all_i)} unique pairs exist")

    choice = rng.choice(len(all_i), size=N_PAIRWISE, replace=False)
    to_rating, sim_min, sim_max = _rating_mapper(s_train)
    real = []
    scrambled = []
    metadata = []

    for pos in choice:
        i = int(all_i[pos])
        j = int(all_j[pos])
        if rng.random() < 0.5:
            i, j = j, i
        a = concepts[i]
        b = concepts[j]
        prompt = pairwise_prompt(a, b)
        real_label = to_rating(s_train[i, j])
        scrambled_label = to_rating(s_scrambled[i, j])
        real.append(_chat_example(prompt, real_label))
        scrambled.append(_chat_example(prompt, scrambled_label))
        metadata.append(
            {
                "view": "pairwise",
                "concepts": (a, b),
                "real_label": str(real_label),
                "scrambled_label": str(scrambled_label),
            }
        )

    return real, scrambled, metadata, sim_min, sim_max


def _sample_feature_bucket(rng, bucket, n_samples):
    replace = len(bucket) < n_samples
    choice = rng.choice(len(bucket), size=n_samples, replace=replace)
    return bucket[choice]


def _generate_features(rng, m_train, m_scrambled, feature_names, concepts):
    if N_FEATURE % 4 != 0:
        raise ValueError("N_FEATURE must be divisible by 4 for balanced real/scrambled labels")

    real = []
    scrambled = []
    metadata = []
    per_bucket = N_FEATURE // 4
    buckets = {}
    for real_value in (0, 1):
        for scrambled_value in (0, 1):
            rows, cols = np.nonzero((m_train == real_value) & (m_scrambled == scrambled_value))
            bucket = np.column_stack([rows, cols])
            if len(bucket) == 0:
                raise ValueError(
                    f"No feature pairs for real={real_value}, scrambled={scrambled_value}"
                )
            buckets[(real_value, scrambled_value)] = bucket

    samples = []
    for key, bucket in buckets.items():
        samples.append(_sample_feature_bucket(rng, bucket, per_bucket))
    samples = np.vstack(samples)
    rng.shuffle(samples)

    for concept_i, feature_i in samples:
        concept_i = int(concept_i)
        feature_i = int(feature_i)
        concept = concepts[concept_i]
        feature = feature_names[feature_i]
        real_value = int(m_train[concept_i, feature_i])
        scrambled_value = int(m_scrambled[concept_i, feature_i])
        prompt = feature_prompt(feature, concept)
        real_label = "True" if real_value else "False"
        scrambled_label = "True" if scrambled_value else "False"
        real.append(_chat_example(prompt, real_label))
        scrambled.append(_chat_example(prompt, scrambled_label))
        metadata.append(
            {
                "view": "feature",
                "concepts": (concept,),
                "feature": feature,
                "real_label": real_label,
                "scrambled_label": scrambled_label,
            }
        )

    return real, scrambled, metadata, {k: len(v) for k, v in buckets.items()}


def _assert_no_test_leakage(metadata, train, test):
    train_set = set(train)
    test_set = set(test)
    leaked = []
    not_train = []
    for row in metadata:
        for concept in row["concepts"]:
            if concept in test_set:
                leaked.append((row["view"], concept))
            if concept not in train_set:
                not_train.append((row["view"], concept))
    if leaked:
        raise ValueError(f"Test concept leakage in generated examples: {leaked[:20]}")
    if not_train:
        raise ValueError(f"Non-train concept in generated examples: {not_train[:20]}")


def _assert_no_test_strings_in_prompts(examples, test):
    patterns = _concept_patterns(test)
    hits = []
    for line_no, example in enumerate(examples, 1):
        prompt = example["messages"][0]["content"].lower()
        for concept, pattern in patterns:
            if pattern.search(prompt):
                hits.append((line_no, concept, prompt[:200]))
                break
        if len(hits) >= 10:
            break
    if hits:
        raise ValueError(f"Test concept strings found in prompts: {hits}")


def _write_jsonl(path, examples):
    with open(path, "w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def _count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def _print_examples(title, examples, n=3):
    print(title)
    for example in examples[:n]:
        user = example["messages"][0]["content"]
        assistant = example["messages"][1]["content"]
        print(f"  user: {user}")
        print(f"  assistant: {assistant}")


def main():
    rng = np.random.default_rng(SEED)
    s_star, s_train, m_train, feature_names, excluded_features, train, test = _load_inputs()
    perm = rng.permutation(len(train))
    s_scrambled = s_train[np.ix_(perm, perm)]
    m_scrambled = m_train[perm]

    triplet_real, triplet_scrambled, triplet_meta = _generate_triplets(
        rng, s_train, s_scrambled, train
    )
    pair_real, pair_scrambled, pair_meta, sim_min, sim_max = _generate_pairwise(
        rng, s_train, s_scrambled, train
    )
    feature_real, feature_scrambled, feature_meta, feature_bucket_counts = _generate_features(
        rng, m_train, m_scrambled, feature_names, train
    )

    paired = list(
        zip(
            triplet_real + pair_real + feature_real,
            triplet_scrambled + pair_scrambled + feature_scrambled,
            triplet_meta + pair_meta + feature_meta,
        )
    )
    _assert_no_test_leakage([row[2] for row in paired], train, test)
    rng.shuffle(paired)

    real_examples = [row[0] for row in paired]
    scrambled_examples = [row[1] for row in paired]
    metadata = [row[2] for row in paired]
    _assert_no_test_strings_in_prompts(real_examples, test)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    _write_jsonl(OUT_PATH, real_examples)
    _write_jsonl(SCRAMBLED_OUT_PATH, scrambled_examples)

    if not os.path.exists(OUT_PATH) or not os.path.exists(SCRAMBLED_OUT_PATH):
        raise FileNotFoundError("Expected JSONL outputs were not written")
    if _count_lines(OUT_PATH) != len(real_examples):
        raise ValueError("Real JSONL line count mismatch")
    if _count_lines(SCRAMBLED_OUT_PATH) != len(scrambled_examples):
        raise ValueError("Scrambled JSONL line count mismatch")

    counts = Counter(row["view"] for row in metadata)
    triplet_modes = Counter(row["mode"] for row in metadata if row["view"] == "triplet")
    real_labels = Counter(row["real_label"] for row in metadata)
    scrambled_labels = Counter(row["scrambled_label"] for row in metadata)

    print("SFT data generation complete")
    print(f"split: train={len(train)}, test={len(test)}, S*={s_star.shape[0]}x{s_star.shape[1]}")
    print(
        "counts: "
        f"triplet={counts['triplet']}, pairwise={counts['pairwise']}, "
        f"feature={counts['feature']}, total={len(real_examples)}"
    )
    print(
        "triplet sampling: "
        f"within={triplet_modes['within']}, across={triplet_modes['across']}, "
        f"clusters={min(N_CLUSTERS, len(train))}"
    )
    print(f"pairwise rating scale source: train off-diagonal S* min={sim_min:.4f}, max={sim_max:.4f}")
    print(f"feature bucket availability (real, scrambled): {feature_bucket_counts}")
    print(
        "feature label counts: "
        f"real True={real_labels['True']}, real False={real_labels['False']}, "
        f"scrambled True={scrambled_labels['True']}, "
        f"scrambled False={scrambled_labels['False']}"
    )
    print(
        "feature leakage filter: "
        f"kept={len(feature_names)}, excluded={len(excluded_features)}, "
        f"excluded_examples={excluded_features[:5]}"
    )
    print(f"jsonl lines: train={_count_lines(OUT_PATH)}, scrambled={_count_lines(SCRAMBLED_OUT_PATH)}")
    print("sanity: all prompt concept slots are train concepts; raw test concept prompt hits = 0")
    print(f"wrote: {os.path.relpath(OUT_PATH, ROOT)}")
    print(f"wrote: {os.path.relpath(SCRAMBLED_OUT_PATH, ROOT)}")
    _print_examples("examples from train.jsonl:", real_examples)
    _print_examples("examples from train_scrambled.jsonl:", scrambled_examples)


if __name__ == "__main__":
    main()
