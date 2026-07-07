# n=60 scale point (nested subset of 128)

triplet~feature coherence (r^2, RDM-direct):
| model | n=30 | n=60 | n=128 |
|---|---|---|---|
| HUMAN        | 0.93 | 0.76 | 0.77 |
| llama-3.1-8b | 0.82 | 0.35 | 0.32 |
| olmo2-7b     | 0.80 | 0.17 | 0.23 |
| qwen2.5-7b   | 0.69 | 0.14 | 0.13 |

FINDING: the collapse is a THRESHOLD effect, essentially complete by n=60 (n60~=n128 for
all). 30 concepts = 2 clusters (reptile/tool) is uniquely easy and inflates model coherence;
by 60 diverse concepts the model<->human gap is fully open and stable. Humans drop modestly
(0.93->0.76) and hold; models drop by half-to-two-thirds and stay down.
(qwen2.5-32b @ n60/n128 pending H100 auth.)
