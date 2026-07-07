# CODEX TASK 2: generate mutually-consistent SFT data from S*

Prereq: task 1 done (data/sft/S_star.npy, S_star_concepts.csv, train_concepts.csv=~656,
test_concepts.csv=128, zero leakage). Read research/PLAN.md step 2 and src/prompts.py.
Work on coherence-sft branch. NO GPU. env: coherence (see task1 brief).

## Write src/sft/gen_sft_data.py
From S* (concept-concept similarity) over TRAIN concepts only, generate 3 mutually-consistent views,
all derived from the SAME S* so they agree by construction:
1. triplet (~60k): sample (anchor A, B, C) from train concepts; "more similar to A" = argmax S*(A,*)
   among {B,C}. Multi-abstraction: sample some triplets within-cluster and some across-cluster
   (kmeans S* into ~15 clusters). Prompt = prompts.triplet_prompt(A,B,C); response = chosen concept.
2. pairwise (~30k): sample pairs; map S* -> 1..7 (linear, 7=most similar). Prompt =
   prompts.pairwise_prompt(a,b); response = the integer.
3. feature (~20k): use the NOVA matrix (data/nova/...parquet) restricted to TRAIN concepts; sample
   (concept, feature) pairs balanced True/False from the binary matrix. Prompt =
   prompts.feature_prompt(feat, concept); response = "True"/"False".
Format each as a chat example {"messages":[{"role":"user","content":prompt},
{"role":"assistant","content":response}]}. Mix ~50/25/25. Shuffle (seed 0). Write
data/sft/train.jsonl.
ALSO write data/sft/train_scrambled.jsonl: identical prompts but LABELS from a PERMUTED S*
(shuffle the concept index of S* before labeling) - the delusive control.

## Run + verify
python src/sft/gen_sft_data.py ; assert both jsonl exist, print counts per view + a few examples,
assert no TEST concept appears in any training example.

## Done
Commit BOTH task 1 outputs (if not already) AND task 2 to coherence-sft:
"SFT step 1+2: S*+split, and mutually-consistent triplet/pairwise/feature SFT data (+scrambled control)".
Do NOT push. Do NOT fabricate. Match repo style.
