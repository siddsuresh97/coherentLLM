# CODEX TASK 1: build S* + train/test concept split from NOVA

Context: read research/PLAN.md and research/STATUS.md first. This is step 1 of the coherence-SFT
experiment. Cheap, deterministic, NO GPU. Work on the `coherence-sft` branch.

## Inputs
- NOVA verified concept x feature matrix (786 concepts), staged IN-REPO (gitignored):
  data/nova/verified_matrix_cogsci2025.parquet
  (index = concept names lowercased; columns = features; values binary 0/1). Needs pyarrow (installed
  in env /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence).
- EVAL concepts (the fixed 128 THINGS test set): data/scale128/concepts.csv (+ data/scale60,
  data/stimuli/concepts.csv = 30, all nested/relevant).

## Write: src/sft/build_target.py
1. Load NOVA parquet. Lowercase/strip the concept index.
2. Compute S* = cosine similarity over the binary feature rows -> 786 x 786 concept-concept matrix.
   Save data/sft/S_star.npy AND data/sft/S_star_concepts.csv (row order).
3. Build the split:
   - test = the 128 eval concepts (data/scale128/concepts.csv). Report how many are in NOVA;
     list any missing.
   - train = ALL OTHER NOVA concepts, EXCLUDING the 128 test AND their close synonyms/hypernyms
     to avoid leakage. For synonym exclusion: at minimum exclude exact-match and simple morphological
     variants; if a wordnet/synonym resource is easily available use it, else do a conservative
     string-similarity exclusion and LOG what was excluded. Target ~650 train concepts.
   - Save data/sft/train_concepts.csv, data/sft/test_concepts.csv.
4. Print a summary: |NOVA|, |test in NOVA|, |train|, #excluded-as-leakage, and a few sample rows.
5. Sanity assert: train ∩ test == empty (incl synonyms you excluded).

## Run it
Activate env: source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh &&
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
Run: python src/sft/build_target.py   (from repo root)

## Done criteria
- data/sft/{S_star.npy, S_star_concepts.csv, train_concepts.csv, test_concepts.csv} exist.
- Summary printed; zero train/test leakage.
- Commit to coherence-sft: "SFT step 1: build S* + train(~650)/test(128) split from NOVA".
- Do NOT push. Do NOT fabricate data. Match existing repo style (see src/analysis.py).
