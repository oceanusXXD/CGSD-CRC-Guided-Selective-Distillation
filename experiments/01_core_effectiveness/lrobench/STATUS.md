# LROBench Core Effectiveness Status

Source run: `experiments/runs/lrobench_ordered`.

Included artifacts:
- `table1_zero_shot_vs_dbds60.csv`: zero-shot + CRC vs DBDS-60 SFT + CRC.
- `zero_shot_round0_summary.json`: round 0 pooled CRC summary.
- `dbds60_round1_summary.json`: DBDS-60 round 1 pooled CRC summary.

Scope: LROBench pooled ordered run only. FEVER is not included.

Important result: DBDS-60 did not reduce defer rate in this run; round 1 defers all pool samples.
