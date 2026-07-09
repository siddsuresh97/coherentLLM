# Literature Note: Huth, Fedorenko, and Cross-Modal Concepts

## Huth / LeBel Narrative Encoding

Sources:

- Huth et al. 2016 semantic maps:
  https://www.nature.com/articles/nature17637
- LeBel et al. 2023 natural-language fMRI dataset:
  https://www.nature.com/articles/s41597-023-02437-z
- Antonello, Vaidya, and Huth scaling laws:
  https://arxiv.org/abs/2305.11863

Method extraction:

- participants listen to natural narrative speech;
- stimulus features are aligned to fMRI TRs;
- delayed feature matrices model the hemodynamic response;
- voxelwise ridge predicts held-out BOLD;
- performance is held-out correlation, ideally noise-ceiling-aware.

Local adaptation:

- use staged `ds003020-smoke` first;
- score `base`, aligned arms, and `scrambled` under the same ridge setup;
- compare paired deltas vs base by subject, story, layer, and ROI;
- only scale to high-data staging after smoke scoring passes.

## Pereira / Ryskina / Fedorenko Cross-Modal Concepts

Sources:

- Pereira et al. 2018 universal decoder:
  https://www.nature.com/articles/s41467-018-03068-4
- Ryskina et al. 2025 concept consistency:
  https://arxiv.org/abs/2508.11536

Method extraction:

- 180 concepts;
- each concept appears through sentences, word clouds, and pictures;
- semantic consistency measures whether a voxel/region responds similarly to
  the same concept across paradigms;
- model-brain alignment is tested by ridge encoding and concept-level RSA.

Local adaptation:

- start with text-only sentence and word-cloud spokes;
- treat picture labels/captions as exploratory proxy unless a VLM/image encoder
  is added;
- test whether coherence-SFT improves alignment with high-semantic-consistency
  regions more than with merely language-selective or visual regions.

## Fedorenko-Style Constraints

Sources:

- Formal vs functional competence in LLMs:
  https://arxiv.org/abs/2301.06627
- Model-internal language network localization:
  https://arxiv.org/abs/2411.02280

Operational constraints:

- use subject-specific language fROIs when making language-network claims;
- label atlas/broad ROI effects exploratory;
- separate formal language, semantic memory, reasoning, and calibration;
- interpret benchmark drops skill-by-skill instead of treating them as one
  global capability loss.

Local model-internal analogue:

- localize sentence-selective or semantic-selective units/layers;
- compare their overlap with concept-vector steering layers;
- ablate/steer and measure benchmark damage by skill family.
