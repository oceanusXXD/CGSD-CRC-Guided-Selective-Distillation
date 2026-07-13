# Multiclass and Preference Sampling-Shift Protocol

## Current Prepared Data

- AG News: `experiments/inputs/benchmarks/ag_news/train.jsonl` has 120,000 rows, 30,000 per class; `test.jsonl` has 7,600 rows, 1,900 per class.
- DBpedia-14: `experiments/inputs/benchmarks/dbpedia_14/train_5000_per_class.jsonl` has 70,000 rows, exactly 5,000 per class; `test.jsonl` is also balanced at 5,000 per class.
- HelpSteer2-Preference: `experiments/inputs/benchmarks/helpsteer2_preference/train.jsonl` has 8,677 pairs and `val.jsonl` has 448 pairs. All 18,250 responses match the five HelpSteer2 attributes.

The checked CPU smoke runs use `Qwen/Qwen3-0.6B` and verify four-class, fourteen-class, and dual-order pairwise probability scoring. The two-pair HelpSteer2 smoke run also shows strong first-position sensitivity, so full diagnostics must retain `order_disagreement` alongside entropy and margin.

All aggregation after model scoring is CPU-only. Classification diagnostics report permutation-corrected uncertainty--group dependence, nested Random trajectories, per-budget and global Random envelopes, centered/positive AAS, and worst-group coverage. Preference diagnostics build separate Random envelopes for all pairs and the non-tie DPO pool across length, attributes, preference direction, prompt distribution, and order disagreement.

## 1. AG News Zero-Shot Diagnostics

```bash
python scripts/benchmark_pipeline.py score-classification \
  --data-path experiments/inputs/benchmarks/ag_news/train.jsonl \
  --output-path experiments/inputs/benchmarks/ag_news/qwen3_0.6b_train_scored.jsonl \
  --model Qwen/Qwen3-0.6B \
  --batch-size 128 \
  --row-batch-size 128 \
  --max-length 1024 \
  --torch-dtype bfloat16 \
  --resume

python scripts/benchmark_pipeline.py diagnose-classification \
  --scored-path experiments/inputs/benchmarks/ag_news/qwen3_0.6b_train_scored.jsonl \
  --output-dir experiments/runs/benchmark_shift/ag_news \
  --budgets 500,1000,2000 \
  --methods random,entropy,margin \
  --seed 42 \
  --random-repetitions 1000 \
  --dependence-permutations 999 \
  --quantile-bins 10
```

Use calibrated category TV, per-class enrichment, adjusted uncertainty coefficient, and worst-group coverage to decide whether entropy or margin is the uncertainty baseline. Prefer the method with the larger stable shift across the three budgets; use entropy if the two are comparable.

```bash
python scripts/benchmark_pipeline.py analyze-shifts \
  --classification-diagnostics experiments/runs/benchmark_shift/ag_news/classification_diagnostics.json \
  --output-path experiments/runs/benchmark_shift/ag_news/shift_gate.json
```

The default gate requires the same budgets to exceed the global Random TV envelope and the class-specific Random enrichment envelope, while retaining minimum effect-size thresholds. Thresholds are CLI-configurable.

## 2. AG News Minimal LoRA

The default one-budget reproduction is 1,000 examples.

```bash
python scripts/benchmark_pipeline.py train-classification-lora \
  --train-path experiments/runs/benchmark_shift/ag_news/random_budget_1000.jsonl \
  --output-dir experiments/runs/benchmark_shift/ag_news/lora_random_1000 \
  --model Qwen/Qwen3-0.6B \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --max-length 1024 \
  --mixed-precision bf16 \
  --dtype bfloat16

python scripts/benchmark_pipeline.py train-classification-lora \
  --train-path experiments/runs/benchmark_shift/ag_news/entropy_budget_1000.jsonl \
  --output-dir experiments/runs/benchmark_shift/ag_news/lora_entropy_1000 \
  --model Qwen/Qwen3-0.6B \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --max-length 1024 \
  --mixed-precision bf16 \
  --dtype bfloat16
```

Evaluate both adapters with `score-classification`, passing the adapter directory through `--model`; local PEFT adapters are detected automatically.

After scoring the same test split with the base, Random-trained, and uncertainty-trained models, generate a directly comparable report:

```bash
python scripts/benchmark_pipeline.py compare-evaluations \
  --task classification \
  --base-path experiments/runs/benchmark_shift/ag_news/eval_base.jsonl \
  --random-path experiments/runs/benchmark_shift/ag_news/eval_random_1000.jsonl \
  --uncertainty-path experiments/runs/benchmark_shift/ag_news/eval_entropy_1000.jsonl \
  --output-path experiments/runs/benchmark_shift/ag_news/model_comparison.json
```

## 3. DBpedia-14 Diagnostics and One-Budget LoRA

```bash
python scripts/benchmark_pipeline.py score-classification \
  --data-path experiments/inputs/benchmarks/dbpedia_14/train_5000_per_class.jsonl \
  --output-path experiments/inputs/benchmarks/dbpedia_14/qwen3_0.6b_train_scored.jsonl \
  --model Qwen/Qwen3-0.6B \
  --batch-size 128 \
  --row-batch-size 64 \
  --max-length 1024 \
  --torch-dtype bfloat16 \
  --resume

python scripts/benchmark_pipeline.py diagnose-classification \
  --scored-path experiments/inputs/benchmarks/dbpedia_14/qwen3_0.6b_train_scored.jsonl \
  --output-dir experiments/runs/benchmark_shift/dbpedia_14 \
  --budgets 500,1000,2000 \
  --methods random,entropy,margin \
  --seed 42 \
  --random-repetitions 1000 \
  --dependence-permutations 999 \
  --quantile-bins 10
```

Repeat the two LoRA commands above at budget 1,000 with the DBpedia selection files. Evaluate on `experiments/inputs/benchmarks/dbpedia_14/test.jsonl`.

## 4. HelpSteer2 Dual-Order Pairwise Diagnostics

```bash
python scripts/benchmark_pipeline.py score-preference \
  --data-path experiments/inputs/benchmarks/helpsteer2_preference/train.jsonl \
  --output-path experiments/inputs/benchmarks/helpsteer2_preference/qwen3_0.6b_train_scored.jsonl \
  --model Qwen/Qwen3-0.6B \
  --batch-size 64 \
  --row-batch-size 16 \
  --max-length 4096 \
  --torch-dtype bfloat16 \
  --resume

python scripts/benchmark_pipeline.py diagnose-preference \
  --scored-path experiments/inputs/benchmarks/helpsteer2_preference/qwen3_0.6b_train_scored.jsonl \
  --output-dir experiments/runs/benchmark_shift/helpsteer2_preference \
  --budget 1000 \
  --methods random,entropy,margin \
  --seed 42 \
  --random-repetitions 1000
```

The report compares response and preferred/rejected lengths, five response attributes and their winner-minus-loser gaps, human preference direction, prompt token-distribution divergence, entropy, margin, and dual-order disagreement.

```bash
python scripts/benchmark_pipeline.py analyze-shifts \
  --preference-diagnostics experiments/runs/benchmark_shift/helpsteer2_preference/preference_diagnostics.json \
  --output-path experiments/runs/benchmark_shift/helpsteer2_preference/shift_gate.json
```

The DPO gate uses the equal-size non-tie subsets and requires material movement beyond the corresponding Random 95% envelope in at least two of length, attributes, preference direction, and prompt distribution. Position-order disagreement cannot establish data shift; if its pool mean exceeds the reliability threshold, DPO training is blocked until pairwise scoring is repaired or recalibrated.

## 5. Random-DPO vs Uncertainty-DPO

```bash
python scripts/benchmark_pipeline.py train-dpo \
  --train-path experiments/runs/benchmark_shift/helpsteer2_preference/random_dpo_budget_1000.jsonl \
  --output-dir experiments/runs/benchmark_shift/helpsteer2_preference/dpo_random_1000 \
  --model Qwen/Qwen3-0.6B \
  --beta 0.1 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 4096 \
  --mixed-precision bf16 \
  --dtype bfloat16

python scripts/benchmark_pipeline.py train-dpo \
  --train-path experiments/runs/benchmark_shift/helpsteer2_preference/entropy_dpo_budget_1000.jsonl \
  --output-dir experiments/runs/benchmark_shift/helpsteer2_preference/dpo_entropy_1000 \
  --model Qwen/Qwen3-0.6B \
  --beta 0.1 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 4096 \
  --mixed-precision bf16 \
  --dtype bfloat16
```

Score `val.jsonl` with the base model and both adapter directories. Only after the measured classification and DPO shifts are available should PCSS be retained, modified with WSR, or replaced by a new objective.

```bash
python scripts/benchmark_pipeline.py compare-evaluations \
  --task preference \
  --base-path experiments/runs/benchmark_shift/helpsteer2_preference/eval_base.jsonl \
  --random-path experiments/runs/benchmark_shift/helpsteer2_preference/eval_dpo_random_1000.jsonl \
  --uncertainty-path experiments/runs/benchmark_shift/helpsteer2_preference/eval_dpo_entropy_1000.jsonl \
  --output-path experiments/runs/benchmark_shift/helpsteer2_preference/model_comparison.json
```
