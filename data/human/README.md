# Human reference data (needed to finish human alignment)

The coherence pipeline produces two kinds of numbers:

1. **Cross-method coherence** (model-only): works now, no human data needed.
2. **Human alignment**: needs a human reference over the SAME 30 concepts
   (`../stimuli/concepts.csv`, in that exact row order).

## What I need from you

Pick whichever exists:

### A. Human triplet / pairwise judgments on these 30 concepts
If the earlier study collected human similarity judgments on the 15 reptiles +
15 tools, give me either:
- raw human triplet choices (anchor, c1, c2, chosen), or
- a human 30x30 similarity/RDM.

Drop a 30x30 numpy array at `human_similarity.npy` (concepts.csv order) and run:
`python src/compute_coherence.py --human_rdm data/human/human_similarity.npy`

### B. Feature-verification ground-truth matrix
For the **feature** method you mentioned you'd supply "the matrix needed for
verification." I need a concept x feature ground-truth (which of the 764 features
in `../stimuli/features.csv` are actually TRUE for each of the 30 concepts), so I
can score model feature-verification accuracy against ground truth (not just use it
to derive a concept RDM). Format: CSV with concepts as rows, features as columns,
0/1 cells. If the Leuven/McRae norms cover these concepts, point me at the file.

## Interim proxy (in use now)
Until A is provided, `fasttext_*.npy` (FastText cc.en.300 cosine over concept
names) is used as a labeled **semantic proxy** for alignment. It shows clean
reptile/tool structure but is NOT human data. All human-alignment numbers computed
against it are marked `fasttext_*` in the output, not `human_*`.
