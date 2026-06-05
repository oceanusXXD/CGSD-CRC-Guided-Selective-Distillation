# Anonymous Binary Selection Code

This folder contains the anonymized code package for a generic binary
query-document selection pipeline. It is a copied package; the original working
tree remains unchanged.

No datasets, model checkpoints, experiment outputs, result tables, logs, or
private service endpoints are included.

## Contents

- `src/`: reusable library code for data loading, CRC calibration, selection,
  metrics, LoRA training, and local evaluation.
- `scripts/`: command-line entrypoints for data conversion, split preparation,
  prediction, CRC computation, selection, and training.
- `requirements.txt`: Python dependencies.

## Path Policy

All examples use relative paths. Suggested local directories are:

- `data/`: external input data prepared by the user.
- `runs/`: generated intermediate artifacts and checkpoints.
- `models/`: local base models or embedding models.

These directories are intentionally not included in this package.

## Input Format

The pipeline expects JSONL rows with a stable id, query, document, and binary
label:

```json
{"id":"sample_1","query":"Does the document satisfy the condition?","document":"...","groundtruth":1}
```

Labels must use the shared binary protocol:

- `1`: the document satisfies the query.
- `0`: the document does not satisfy the query.

## Minimal Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Convert external JSONL into the canonical format:

```bash
python scripts/convert_jsonl.py \
  --input_path data/raw/dataset_a.jsonl \
  --output_path data/dataset_a/data.jsonl \
  --id_field id \
  --query_field query \
  --document_field document \
  --label_field label
```

Create guide, final, and pool ids:

```bash
python scripts/prepare.py \
  --data_path data/dataset_a/data.jsonl \
  --output_dir runs/dataset_a/run_a \
  --n_guide 1000 \
  --n_final 200 \
  --seed 1 \
  --cache_policy overwrite
```

Run base-model prediction through an OpenAI-compatible local server:

```bash
python scripts/predict_vllm_openai.py \
  --output_dir runs/dataset_a/run_a \
  --round_index 0 \
  --model_path models/base-model \
  --data_path data/dataset_a/data.jsonl \
  --split_ids_path runs/dataset_a/run_a/split_ids.json \
  --temperature 0.0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --start_server \
  --cache_policy overwrite
```

Compute CRC decisions:

```bash
python scripts/compute_crc.py \
  --output_dir runs/dataset_a/run_a \
  --round_index 0 \
  --temperature 1.0 \
  --alpha 0.1 \
  --cache_policy overwrite
```

Select training rows:

```bash
python scripts/select_pcss.py \
  --output_dir runs/dataset_a/run_a \
  --round_index 0 \
  --budget 250 \
  --seed 1 \
  --cache_policy overwrite
```

Train a LoRA round:

```bash
python scripts/train_round.py \
  --output_dir runs/dataset_a/run_a \
  --round_index 1 \
  --model_path models/base-model \
  --data_path data/dataset_a/data.jsonl \
  --split_ids_path runs/dataset_a/run_a/split_ids.json \
  --train_rows_path runs/dataset_a/run_a/train_rows.jsonl \
  --epochs 4 \
  --cache_policy overwrite
```

Inference after training can use either local PyTorch scoring or an
OpenAI-compatible vLLM server. Both paths use the shared `chat_binary` prompt,
the same `1`-minus-`0` score definition, and the same query, document, label,
model, data, and split-id inputs.

Local PyTorch inference:

```bash
python scripts/predict_local.py \
  --checkpoint_dir runs/dataset_a/run_a/round_1/model \
  --model_path models/base-model \
  --data_path data/dataset_a/data.jsonl \
  --split_ids_path runs/dataset_a/run_a/split_ids.json \
  --split_name pool \
  --threshold 0.0 \
  --predictions_path runs/dataset_a/run_a/round_1/pool_predictions.jsonl \
  --metrics_path runs/dataset_a/run_a/round_1/pool_metrics.json
```

vLLM inference:

```bash
python scripts/predict_vllm_openai.py \
  --output_dir runs/dataset_a/run_a \
  --round_index 1 \
  --checkpoint_dir runs/dataset_a/run_a/round_1/model \
  --model_path models/base-model \
  --data_path data/dataset_a/data.jsonl \
  --split_ids_path runs/dataset_a/run_a/split_ids.json \
  --temperature 0.0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --start_server \
  --cache_policy overwrite
```

## Selection Methods

- `select_random.py`: uniform sampling from the candidate pool.
- `select_pcss.py`: prior-corrective stratified selection. The target positive
  rate is estimated from guide labels, then enforced on candidate-pool proxy
  predictions while prioritizing uncertain samples within each proxy-label
  stratum.
- `select_crc_error_mass.py`: CRC error-mass budget split between accept and
  defer sides.
