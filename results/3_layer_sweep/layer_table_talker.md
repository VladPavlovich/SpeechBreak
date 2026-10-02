# Step 3 — Talker layer sweep

Harmful − harmless direction (at the last audio token), added raw (α = 1) at every position of one Talker layer; one forward pass per (prompt, layer), no generation. 8 harmless prompts get +direction, 8 harmful prompts get −direction, all from the val split.

- **sep d / acc**: held-out Cohen's d and balanced accuracy of the projection on the direction
- **KL**: KL(unsteered ‖ steered) of the first codec-token distribution (the Talker's first audio token; the Thinker's text is unchanged) (nats)
- **top-1 Δ**: fraction of prompts whose most likely first token changed

Ranked by **kl** among layers 4–19 with sep d ≥ 1. **Top layers: [5, 4]**

| layer | |dir|/|resid| | sep d | sep acc | KL | KL harmless | KL harmful | top-1 Δ |  |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.329 | 2.01 | 0.911 | 0.177 | 0.177 | 0.176 | 0.06 | out of band |
| 1 | 0.305 | 1.98 | 0.861 | 0.145 | 0.154 | 0.135 | 0.00 | out of band |
| 2 | 0.233 | 1.95 | 0.851 | 0.040 | 0.050 | 0.031 | 0.00 | out of band |
| 3 | 0.199 | 1.68 | 0.804 | 0.017 | 0.021 | 0.012 | 0.00 | out of band |
| 4 | 0.167 | 1.32 | 0.734 | 0.034 | 0.044 | 0.023 | 0.00 | **top** |
| 5 | 0.163 | 1.13 | 0.699 | 0.042 | 0.053 | 0.031 | 0.00 | **top** |
| 6 | 0.139 | 0.87 | 0.673 | 0.018 | 0.019 | 0.016 | 0.00 |  |
| 7 | 0.131 | 0.64 | 0.655 | 0.006 | 0.007 | 0.006 | 0.00 |  |
| 8 | 0.153 | 0.54 | 0.599 | 0.007 | 0.007 | 0.007 | 0.00 |  |
| 9 | 0.156 | 0.49 | 0.599 | 0.009 | 0.008 | 0.009 | 0.00 |  |
| 10 | 0.160 | 0.55 | 0.602 | 0.006 | 0.007 | 0.006 | 0.00 |  |
| 11 | 0.140 | 0.52 | 0.612 | 0.005 | 0.004 | 0.007 | 0.00 |  |
| 12 | 0.131 | 0.52 | 0.599 | 0.005 | 0.004 | 0.007 | 0.00 |  |
| 13 | 0.128 | 0.50 | 0.591 | 0.013 | 0.010 | 0.016 | 0.00 |  |
| 14 | 0.127 | 0.49 | 0.584 | 0.009 | 0.004 | 0.013 | 0.00 |  |
| 15 | 0.129 | 0.51 | 0.579 | 0.023 | 0.017 | 0.028 | 0.00 |  |
| 16 | 0.133 | 0.53 | 0.608 | 0.017 | 0.018 | 0.016 | 0.00 |  |
| 17 | 0.134 | 0.53 | 0.595 | 0.029 | 0.024 | 0.035 | 0.00 |  |
| 18 | 0.130 | 0.57 | 0.590 | 0.041 | 0.049 | 0.033 | 0.00 |  |
| 19 | 0.126 | 0.60 | 0.606 | 0.033 | 0.038 | 0.029 | 0.00 |  |
| 20 | 0.123 | 0.61 | 0.625 | 0.026 | 0.026 | 0.026 | 0.00 | out of band |
| 21 | 0.122 | 0.61 | 0.606 | 0.005 | 0.003 | 0.007 | 0.00 | out of band |
| 22 | 0.122 | 0.62 | 0.608 | 0.005 | 0.005 | 0.005 | 0.00 | out of band |
| 23 | 0.121 | 0.63 | 0.611 | 0.002 | 0.002 | 0.002 | 0.00 | out of band |
