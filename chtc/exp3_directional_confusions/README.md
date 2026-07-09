# Experiment 3 CHTC Runner

This is a tiny GPU runner for Experiment 3 Step 1. It only runs the neutral
geometry/item pipeline; it does not create Step 2 safety items.

## Prepare Bundle

Run from the repository root before submitting:

```bash
bash chtc/exp3_directional_confusions/prepare_exp3_bundle.sh
```

The bundle includes the Experiment 3 scaffold, the runner script, prompt helper,
and Leuven feature files needed to generate neutral items.

## Geometry Job

Submit from a CHTC Access Point after copying this directory and the bundle:

```bash
mkdir -p logs
condor_submit exp3_geometry.sub
```

The geometry job runs the three frozen triplet passes, builds `rdm.npy`,
registers nearest-neighbor predictions, and generates Step 1 items. It stops
before the human sanity gate and before scoring H1.

Expected staged model path:

```text
/staging/s/suresh27/models/llama31-8b-instruct
```

After completion, pull `exp3_geometry_<RUN_ID>_results.tgz`, inspect
`neighbors.json`, mark the sanity gate locally, then run the item pass.

## Item Job

The item job is for after geometry outputs have been pulled into the repo and the
sanity gate has been marked. Rebuild the bundle so it includes `neighbors.json`,
`artifacts/rdm.npy`, and generated items, then set `MODE = items` in a copied
submit file or override it before submission.
