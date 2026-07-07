"""Stage 2: parse a model's feature LISTING into its own concept x feature union.

Reads results/raw/<model>/listing.csv (concept, rep, response), splits each
generation into individual features, normalizes them, and builds:
  - results/raw/<model>/listed_features.csv : the model's unique feature vocabulary
  - results/raw/<model>/verify_pairs.csv    : (feature, concept) pairs to verify.

Which pairs to verify: a model listed feature F for concept C in stage 1, so C-F is
a candidate TRUE. To get a proper True/False signal (and let the model reject its own
noisy listings), we verify every listed feature against EVERY concept by default
(sparse-but-complete), or only the concepts that co-listed it with --listed_only.
"""
import argparse
import csv
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("COHERENCE_RAW_DIR", os.path.join(HERE, "results", "raw"))


def load_concepts():
    stim = os.environ.get("COHERENCE_STIM_DIR", os.path.join(HERE, "data", "stimuli"))
    with open(os.path.join(stim, "concepts.csv")) as f:
        return [ln.strip() for ln in f if ln.strip()]


STOP_PREFIXES = ("here are", "sure", "the features", "properties of", "certainly",
                 "as an", "note:", "these are")


def clean_feature(line: str) -> str:
    s = line.strip()
    # strip bullets / numbering / markdown
    s = re.sub(r"^[\-\*•\d\.\)\(\s]+", "", s)
    s = s.strip(" .;:-\t").lower()
    # drop obvious non-features
    if len(s) < 3 or len(s) > 60:
        return ""
    if any(s.startswith(p) for p in STOP_PREFIXES):
        return ""
    if s.endswith(":"):
        return ""
    return s


def parse_listing(model):
    path = os.path.join(RAW, model, "listing.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # concept -> set of listed features (across reps)
    per_concept = {}
    for _, row in df.iterrows():
        concept = row["concept"]
        for line in str(row["response"]).splitlines():
            feat = clean_feature(line)
            if feat:
                per_concept.setdefault(concept, {})
                per_concept[concept][feat] = per_concept[concept].get(feat, 0) + 1
    return per_concept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--min_listings", type=int, default=1,
                    help="keep a feature for a concept only if listed >= this many reps")
    ap.add_argument("--min_concepts", type=int, default=1,
                    help="keep a feature in the union only if it was listed for >= this "
                         "many concepts (drops one-off noise)")
    ap.add_argument("--listed_only", action="store_true",
                    help="verify each feature only against concepts that listed it "
                         "(else verify against all 30)")
    args = ap.parse_args()

    per_concept = parse_listing(args.model)
    if per_concept is None:
        raise SystemExit(f"no listing.csv for {args.model}")
    concepts = load_concepts()

    # count in how many concepts each feature appears (after min_listings filter)
    feat_concepts = {}
    for c in concepts:
        for feat, n in per_concept.get(c, {}).items():
            if n >= args.min_listings:
                feat_concepts.setdefault(feat, set()).add(c)
    union = sorted(f for f, cs in feat_concepts.items()
                   if len(cs) >= args.min_concepts)

    outdir = os.path.join(RAW, args.model)
    pd.Series(union).to_csv(os.path.join(outdir, "listed_features.csv"),
                            index=False, header=False)

    # build verify pairs
    rows = []
    for feat in union:
        targets = feat_concepts[feat] if args.listed_only else concepts
        for c in targets:
            rows.append((feat, c))
    with open(os.path.join(outdir, "verify_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for feat, c in rows:
            w.writerow([feat, c])

    print(f"[{args.model}] listed vocab={len(union)} features "
          f"(min_listings={args.min_listings}, min_concepts={args.min_concepts}); "
          f"verify pairs={len(rows)} ({'listed-only' if args.listed_only else 'x all 30'})")


if __name__ == "__main__":
    main()
