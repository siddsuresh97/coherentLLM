# Experiment 1 CHTC Pathway Check

This directory prepares the first real GPU run for Experiment 1:
`concentrated_drop_100` with feature-listing supervision.

The job runs the hard gate first. It collects:

- `base_seed_a_canonical_prompt`
- `base_seed_b_canonical_prompt`
- `base_seed_a_paraphrase_prompt`

Then it reruns `scripts/run_experiment1.py`; if `floor_stats.json` is not green,
the job stops before LoRA training. If the gate passes, it trains control/edit
LoRAs, recovers triplet RDMs, scores detection, and writes a result bundle.

## Prepare Bundle Locally

From the repo root:

```bash
chmod +x chtc/exp1_triplet_move/prepare_exp1_bundle.sh
chtc/exp1_triplet_move/prepare_exp1_bundle.sh
```

This writes:

```text
chtc/exp1_triplet_move/exp1_triplet_move_bundle.tgz
```

## Submit On CHTC

Use an authenticated CHTC SSH master first. BatchMode SSH from this environment
is denied without Duo/interactive auth.

```bash
# In your terminal, once per session:
chtc-master start
chtc-master check
chtc-ssh 'hostname -f && condor_q && echo "$STAGING"'
```

Then push and submit:

```bash
chtc/exp1_triplet_move/submit_exp1_pathway.sh
```

Manual equivalent:

```bash
chtc-ssh 'mkdir -p ~/chtc-runs/exp1-triplet-move-pathway/logs'
chtc-push chtc/exp1_triplet_move/run_exp1_pathway_check.sh 'chtc-runs/exp1-triplet-move-pathway/'
chtc-push chtc/exp1_triplet_move/exp1_pathway_check.sub 'chtc-runs/exp1-triplet-move-pathway/'
chtc-push chtc/exp1_triplet_move/exp1_triplet_move_bundle.tgz 'chtc-runs/exp1-triplet-move-pathway/'
chtc-ssh 'cd ~/chtc-runs/exp1-triplet-move-pathway && condor_submit exp1_pathway_check.sub'
```

Monitor:

```bash
chtc-ssh 'condor_q -nobatch'
chtc-ssh 'cd ~/chtc-runs/exp1-triplet-move-pathway && ls -lh logs'
```

Pull results:

```bash
mkdir -p chtc-runs/exp1-triplet-move-pathway
chtc-pull 'chtc-runs/exp1-triplet-move-pathway/' chtc-runs/exp1-triplet-move-pathway/
```

Expected output bundle:

```text
exp1_pathway_concentrated_drop_100_feature_results.tgz
```

## Switching To The Fallback Lever

If the feature-listing pathway check does not move the behavioral triplet RDM,
submit the same job with direct pairwise similarity supervision:

```bash
SUPERVISION=similarity chtc/exp1_triplet_move/submit_exp1_pathway.sh
```

That still evaluates on the held-out frozen triplet protocol.
