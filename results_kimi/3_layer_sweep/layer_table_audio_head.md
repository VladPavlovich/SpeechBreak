# Step 3 — Kimi-Audio Audio-head layer sweep

Harmful − harmless direction (at the last prompt token), added raw (α = 1) at every position of one Audio-head layer; forward passes only, no generation. 8 harmless prompts get +direction, 8 harmful prompts get −direction, all from the val split.

- **sep d / acc**: held-out Cohen's d and balanced accuracy of the projection on the direction
- **KL**: KL(unsteered ‖ steered) of the first audio-token distribution (the audio stream's first token, 6 steps after the text starts; the text is unchanged) (nats)
- **top-1 Δ**: fraction of prompts whose most likely first token changed

Ranked by **kl** among layers 1–4 with sep d ≥ 1. **Top layers: [1, 2, 3]**

| layer | |dir|/|resid| | sep d | sep acc | KL | KL harmless | KL harmful | top-1 Δ |  |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.399 | 4.72 | 0.997 | 0.219 | 0.278 | 0.160 | 0.31 | out of band |
| 1 | 0.330 | 4.68 | 0.995 | 0.454 | 0.322 | 0.585 | 0.38 | **top** |
| 2 | 0.328 | 4.40 | 0.992 | 0.376 | 0.334 | 0.418 | 0.50 | **top** |
| 3 | 0.319 | 3.79 | 0.987 | 0.269 | 0.175 | 0.362 | 0.44 | **top** |
| 4 | 0.330 | 3.63 | 0.982 | 0.146 | 0.128 | 0.164 | 0.31 |  |
| 5 | 0.325 | 3.48 | 0.966 | 0.179 | 0.209 | 0.149 | 0.38 | out of band |
