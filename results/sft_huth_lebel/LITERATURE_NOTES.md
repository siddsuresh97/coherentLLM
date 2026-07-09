# Literature Notes For Huth/LeBel, Fedorenko, And Semantic-Hub Evaluation

Created: 2026-07-08.

Scope: concise literature synthesis for designing the next coherence-SFT fMRI
and semantic-hub experiments. This file is a handoff note, not a full review.

## Current Repo Context

- The THINGS object-fMRI RSA track is already complete in
  `results/sft_fmri/REPORT.md`. It shows a working RSA pipeline and strong
  scrambled-control separation in Ventral Visual, but little primary Ventral
  Visual gain for aligned arms over base.
- The first internal semantic-hub track is already complete in
  `results/sft_semantic_hub/REPORT.md`; aligned/task-vector arms improve
  cross-format invariance over base.
- The paper-style matched-vs-baseline semantic-hub follow-up is complete in
  `results/sft_semantic_hub_paper/REPORT.md`. Matched concepts beat random
  mismatches more strongly in aligned/task-vector arms, but exact same-concept
  over S*-close-neighbor margins are small.
- The Huth/LeBel language-fMRI lane has staged smoke data on CHTC at
  `/staging/s/suresh27/datasets/ds003020-smoke`, validated by cluster
  `5513059`. The current executable plan is
  `results/sft_huth_lebel/ENCODING_PLAN.md`.

## Source Notes

| Source | Why It Matters Here | Design Implication |
|---|---|---|
| Huth et al. 2016, "Natural speech reveals the semantic maps that tile human cerebral cortex" | Voxelwise encoding during hours of narrative listening produced a data-driven semantic atlas across cortex. | Our Huth-style claim should be held-out voxelwise encoding predictivity for passive story listening, not prompt-task accuracy. Use story text, no chat template, and fit linear/ridge mappings to BOLD. Link: https://www.nature.com/articles/nature17637 |
| LeBel et al. 2023, "A natural language fMRI dataset for voxelwise encoding models" | The `ds003020` dataset contains 8 participants listening to 27 natural narrative stories, about 6 hours each, with raw/preprocessed MRIs and code support. | This is the primary practical dataset for our next language-fMRI lane. Start with staged `UTS01`-`UTS03` smoke files, then scale only after the smoke encoding report passes. Link: https://www.nature.com/articles/s41597-023-02437-z |
| OpenNeuro `ds003020` | Public BIDS/datalad source for the LeBel dataset. | CHTC jobs should rely on staged explicit file manifests or direct containerized downloads. Do not assume local `/mnt/dv` paths exist on execute nodes. Link: https://openneuro.org/datasets/ds003020 |
| HuthLab `deep-fMRI-dataset` code | Official code accompanying LeBel et al.; includes preprocessed data loading and voxelwise encoding scripts. | Our smoke implementation should mirror its structure where feasible: word/story features, delays, ridge, held-out story performance. Link: https://github.com/HuthLab/deep-fMRI-dataset |
| Antonello, Vaidya, Huth 2023, "Scaling laws for language encoding models in fMRI" | Transformer language-model representations predict natural-language fMRI; performance scales with model size and fMRI training data. | We should expect data amount to matter. Smoke results are engineering checks, while high-data `UTS01`-`UTS03` is the first plausible scientific run. Link: https://arxiv.org/abs/2305.11863 |
| HuthLab `encoding-model-scaling-laws` code | Provides feature extraction and encoding-model reference code for the scaling-law work. | Future agents should check this repo before inventing a new ridge/feature-extraction convention for full-scale runs. Link: https://github.com/HuthLab/encoding-model-scaling-laws |
| Popham, Huth et al. 2021, visual and linguistic semantic maps | Visual and linguistic semantic selectivity align along a cortical boundary, suggesting cross-modal semantic organization. | Useful bridge between our THINGS object-fMRI track and narrative language-fMRI. If language-fMRI improves but THINGS Ventral Visual stays flat, test whether gains concentrate in language/semantic-interface regions rather than object visual cortex. Link: https://www.nature.com/articles/s41593-021-00921-6 |
| Lipkin, Tuckute, Ivanova, Fedorenko et al. 2022 language atlas | Fedorenko/EvLab-style language areas are best identified with individual functional localizers; the paper provides a probabilistic atlas from 806 participants for cases where localizers are unavailable. | A true Fedorenko-style claim requires subject-specific language fROIs, typically language > control. If ds003020 lacks localizers in our staged subset, use atlas/LanA labels only as exploratory. Link: https://www.nature.com/articles/s41597-022-01645-3 |
| Fedorenko et al. 2010 language-localizer method | The classic method defines language ROIs functionally in individual subjects. | Do not claim "the language network" from broad anatomical parcels. Report as "atlas language" or "exploratory language-like regions" unless individual localizers are available. DOI link: https://doi.org/10.1152/jn.00032.2010 |
| Mahowald, Ivanova, Blank, Kanwisher, Tenenbaum, Fedorenko 2023 | Distinguishes formal linguistic competence from functional/world cognition and grounds the distinction in human neuroscience. | If coherence-SFT improves semantic/world-concept structure but hurts MMLU/ARC/WiC, the fMRI plan should separate language-network predictivity from broader cognitive/world-knowledge benchmarks. Link: https://arxiv.org/abs/2301.06627 |
| AlKhamissi, Tuckute, Bosselut, Schrimpf 2024, LLM language network | Applies a neuroscience-style sentences > nonwords localizer to LLM units and tests causal relevance. | Optional model-side control: localize Llama units using sentence vs nonword stimuli, then ask whether coherence-SFT changes fMRI predictivity through language-selective units or through a broader semantic hub. Link: https://arxiv.org/abs/2411.02280 |
| Wu, Yu, Yogatama, Lu, Kim 2024/2025, semantic hub hypothesis | Proposes shared intermediate-layer representations across languages/modalities, with relative similarity, logit-lens, and intervention evidence. | Our current triplet/pairwise/feature hub proxy is Stage 0. Stronger evidence needs matched-vs-baseline, logit-lens semantic anchoring, symbolic spokes, and causal patching. Link: https://arxiv.org/abs/2411.04986 |
| Ryskina, Tuckute, Fung, Malkin, Fedorenko 2025 | Connects model-brain alignment to regions that represent concepts consistently across modalities. | This is the most direct adjacent brain-side analogue to semantic-hub memory: concept consistency across sentence, word-cloud, and image paradigms. Use as a conceptual template, not as a ds003020 substitute. Link: https://arxiv.org/abs/2508.11536 |

## Method Lessons To Carry Forward

- Huth/LeBel evaluation is a predictive encoding problem: hidden states at word
  times, hemodynamic delays, voxelwise ridge, held-out story correlation.
- Chat templates are inappropriate for passive listening data. Use plain story
  text and identical transcript reconstruction across all arms.
- Fedorenko/EvLab constraints are about functional specificity. The primary
  localizer contrast is language over degraded controls such as nonword lists;
  broad IFG/temporal anatomy is not enough for a strong claim.
- Semantic-hub evidence needs layer localization. A result only in final layers
  is more likely answer-surface formatting than a shared semantic memory.
- Scrambled is the essential negative control. If `scrambled` improves fMRI or
  hub metrics as much as `lowLR` or `taskvec_a0p25`, the effect is not specific
  to the induced coherence direction.
- `taskvec_a0p25` is the cleanest mitigation arm to compare with `lowLR`: it
  recovers much of the internal semantic-hub gain with less benchmark damage.

## Citation Map For Future Reports

Use these short names consistently:

- `Huth2016`: Huth et al., Nature 2016, semantic maps from natural speech.
- `LeBel2023`: LeBel et al., Scientific Data 2023, `ds003020`.
- `Antonello2023`: Antonello/Vaidya/Huth, NeurIPS 2023, encoding scaling laws.
- `Popham2021`: Popham/Huth et al., Nature Neuroscience 2021, visual-linguistic semantic map alignment.
- `Fedorenko2010`: individual-subject language fROI method.
- `Lipkin2022`: probabilistic language atlas from precision fMRI.
- `Mahowald2023`: language vs thought / formal vs functional competence.
- `AlKhamissi2024`: LLM language-network localizer and ablation.
- `Wu2024`: semantic hub hypothesis, arXiv:2411.04986.
- `Ryskina2025`: concept consistency across modalities and LM-brain alignment.
