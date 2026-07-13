




| --- | ---: | --- | --- | --- | --- | --- |


- `experiments/inputs/imdb/manifest.json`
- `experiments/inputs/imdb/query_id_1/data.jsonl`
- `experiments/inputs/imdb/query_id_1/source.metadata.json`
- `experiments/inputs/imdb/query_id_1/prepare_summary.json`
- `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_split_ids.json`
- `experiments/inputs/imdb/query_id_2/data.jsonl`
- `experiments/inputs/imdb/query_id_2/source.metadata.json`
- `experiments/inputs/imdb/query_id_2/prepare_summary.json`
- `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_split_ids.json`



| query_id | query | rows | label0/no | label1/yes |
| ---: | --- | ---: | ---: | ---: |
| 1 | `Is the following review positive?` | 49990 | 24999 (50.01%) | 24991 (49.99%) |
| 2 | `Does the following review explicitly recommend that others watch the movie?` | 49990 | 36247 (72.51%) | 13743 (27.49%) |


- `source`: `Cascade/datasets/public/imdb/ground_truth.json`
- `documents_source`: `Cascade/datasets/public/imdb/documents.json`
- `queries_source`: `Cascade/datasets/public/imdb/queries.json`
- `overlength_removed_max2048`: 10
- `overlength_removed_manifest`: `outputs/runs/overlength_removed_records_max2048.jsonl`



| ---: | --- | --- | --- | --- |



| --- | --- |
| model path | `/teamspace/studios/this_studio/model/qwen3-4b-embedding` |
| backend | `vllm` |
| request batch size | 256 |
| flush rows | 1024 |
| max length | 4096 |
| torch dtype | `bfloat16` |
| tensor parallel size | 1 |
| gpu memory utilization | 0.92 |
| mode | `document` |



| query_id | split path | guide n | guide label0/no | guide label1/yes | pool n | pool label0/no | pool label1/yes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_split_ids.json` | 1000 | 502 (50.20%) | 498 (49.80%) | 48990 | 24497 (50.00%) | 24493 (50.00%) |
| 2 | `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_split_ids.json` | 1000 | 715 (71.50%) | 285 (28.50%) | 48990 | 35532 (72.53%) | 13458 (27.47%) |


- `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42.ids.json`
- `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42.ids.json`



| query_id | round0 dir | all | guide/calibration | pool | final calibration |
| ---: | --- | --- | --- | --- | --- |


```bash
python scripts/predict_vllm_openai.py \
  --output_dir "experiments/inputs/imdb/query_id_${QID}" \
  --round_index 0 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path "experiments/inputs/imdb/query_id_${QID}/data.jsonl" \
  --split_ids_path "experiments/inputs/imdb/query_id_${QID}/qwen06b_full_guide1000_seed42_split_ids.json" \
  --served_model_name qwen3-0.6b \
  --start_server \
  --base_url http://127.0.0.1:18021/v1 \
  --parallel_requests 8192 \
  --request_retries 3 \
  --timeout 180 \
  --temperature 0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --max_model_len 4096 \
  --max_num_seqs 4096 \
  --max_num_batched_tokens 524288 \
  --gpu_memory_utilization 0.98 \
  --enforce_eager \
  --cache_policy reuse
```


| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |


| query_id | split | n | acc | macro-F1 | F1 no | F1 yes | balanced-acc | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | all | 49990 | 0.5000 | 0.3335 | 0.0003 | 0.6666 | 0.5001 | 24990/4/24995/1 |
| 1 | guide/calibration | 1000 | 0.4980 | 0.3324 | 0.0000 | 0.6649 | 0.5000 | 498/0/502/0 |
| 1 | pool | 48990 | 0.5000 | 0.3335 | 0.0003 | 0.6666 | 0.5001 | 24492/4/24493/1 |
| 2 | all | 49990 | 0.2749 | 0.2157 | 0.0001 | 0.4313 | 0.5000 | 13743/1/36246/0 |
| 2 | guide/calibration | 1000 | 0.2850 | 0.2218 | 0.0000 | 0.4436 | 0.5000 | 285/0/715/0 |
| 2 | pool | 48990 | 0.2747 | 0.2155 | 0.0001 | 0.4310 | 0.5000 | 13458/1/35531/0 |





- `experiments/inputs/imdb/query_id_1/round_0/qwen06b_full_guide1000_seed42_T1_T10_defer_rates_ns.json`
- `experiments/inputs/imdb/query_id_2/round_0/qwen06b_full_guide1000_seed42_T1_T10_defer_rates_ns.json`

q1 T sweep

| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.880589 | 48990 | 28969 | 20021 | 40.87% | 16.02% | 0.5020 | 0.9877 | 0.6727 |
| 2 | 0.802218 | 48990 | 29273 | 19717 | 40.25% | 17.53% | 0.5020 | 0.9782 | 0.6711 |
| 3 | 0.725897 | 48990 | 29321 | 19669 | 40.15% | 17.78% | 0.5020 | 0.9782 | 0.6710 |
| 4 | 0.670746 | 48990 | 29369 | 19621 | 40.05% | 17.78% | 0.5020 | 0.9805 | 0.6711 |
| 5 | 0.631669 | 48990 | 29465 | 19525 | 39.86% | 17.88% | 0.5020 | 0.9829 | 0.6712 |
| 6 | 0.603647 | 48990 | 29477 | 19513 | 39.83% | 17.78% | 0.5020 | 0.9902 | 0.6719 |
| 7 | 0.583817 | 48990 | 29297 | 19693 | 40.20% | 17.19% | 0.5020 | 0.9902 | 0.6722 |
| 8 | 0.567790 | 48990 | 29255 | 19735 | 40.28% | 17.00% | 0.5020 | 0.9902 | 0.6723 |
| 9 | 0.555426 | 48990 | 29201 | 19789 | 40.39% | 16.80% | 0.5020 | 0.9902 | 0.6724 |
| 10 | 0.545307 | 48990 | 29116 | 19874 | 40.57% | 16.51% | 0.5020 | 0.9902 | 0.6726 |

q2 T sweep

| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.010000 | 48990 | 0 | 48990 | 100.00% | 0.00% | 0.7150 | 0.7150 | 1.0000 |
| 2 | 0.942178 | 48990 | 13302 | 35688 | 72.85% | 35.80% | 0.7150 | 0.8567 | 0.7689 |
| 3 | 0.854713 | 48990 | 12656 | 36334 | 74.17% | 36.32% | 0.7150 | 0.8450 | 0.7769 |
| 4 | 0.789216 | 48990 | 12738 | 36252 | 74.00% | 36.58% | 0.7150 | 0.8462 | 0.7759 |
| 5 | 0.742950 | 48990 | 12815 | 36175 | 73.84% | 36.50% | 0.7150 | 0.8485 | 0.7750 |
| 6 | 0.708996 | 48990 | 13003 | 35987 | 73.46% | 36.36% | 0.7150 | 0.8520 | 0.7727 |
| 7 | 0.684469 | 48990 | 12931 | 36059 | 73.60% | 35.90% | 0.7150 | 0.8497 | 0.7734 |
| 8 | 0.665446 | 48990 | 12933 | 36057 | 73.60% | 35.54% | 0.7150 | 0.8497 | 0.7734 |
| 9 | 0.648868 | 48990 | 13430 | 35560 | 72.59% | 35.70% | 0.7150 | 0.8591 | 0.7673 |
| 10 | 0.637072 | 48990 | 13296 | 35694 | 72.86% | 35.41% | 0.7150 | 0.8591 | 0.7693 |






- `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/`
- `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/`


| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2500 | `pool-random` | 2500 | 1499 | 1001 | 40.04% | 1230 | 49.20% | 1231 | 1269 | `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/seed1/pool_random_2500_seed1.summary.json` |
| 1 | 2500 | `ns-error-mass` | 2500 | 819 | 1681 | 67.24% | 2315 | 92.60% | 2316 | 184 | `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/seed1/ns_error_mass_2500_seed1.summary.json` |
| 2 | 2500 | `pool-random` | 2500 | 697 | 1803 | 72.12% | 1805 | 72.20% | 1805 | 695 | `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/seed1/pool_random_2500_seed1.summary.json` |
| 2 | 2500 | `ns-error-mass` | 2500 | 577 | 1923 | 76.92% | 1975 | 79.00% | 1975 | 525 | `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/seed1/ns_error_mass_2500_seed1.summary.json` |


- `experiments/inputs/imdb/query_id_1/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/train2500_two_methods_summary.json`
- `experiments/inputs/imdb/query_id_2/qwen06b_full_guide1000_seed42_T10_alpha010_train2500_two_methods_seed1/train2500_two_methods_summary.json`



| --- | --- |
| base model | `Qwen3-0.6B` |
| model path | `/teamspace/studios/this_studio/model/qwen3-0.6b` |
| input format | `cgsd_chat_binary_v1` |
| round index | 1 |
| epochs | 4 |
| learning rate | `1e-4` |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| target modules | `attention_mlp` |
| lora layer | all |
| max length | 4096 |
| train batch size | 4 |
| gradient accumulation steps | 6 |
| pad to multiple of | 8 |
| seed | 42 |
| precision | `bfloat16` |
| cache policy | `reuse` |


```bash
python scripts/train_round.py \
  --output_dir "$RUN_DIR" \
  --round_index 1 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path "experiments/inputs/imdb/query_id_${QID}/data.jsonl" \
  --split_ids_path "experiments/inputs/imdb/query_id_${QID}/qwen06b_full_guide1000_seed42_split_ids.json" \
  --train_rows_path "$TRAIN_ROWS" \
  --max_length 4096 \
  --epochs 4 \
  --lr 1e-4 \
  --batch_size 4 \
  --gradient_accumulation_steps 6 \
  --pad_to_multiple_of 8 \
  --torch_dtype bfloat16 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules attention_mlp \
  --lora_layer_scope all \
  --seed 42 \
  --cache_policy reuse
```


- `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/`


| ---: | --- | ---: | --- | --- |
| 1 | `pool-random` | 2500 | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q1_pool_random/round_1/model` | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q1_pool_random/round_1/training_round_summary.json` |
| 1 | `ns-error-mass` | 2500 | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q1_ns_error_mass/round_1/model` | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q1_ns_error_mass/round_1/training_round_summary.json` |
| 2 | `pool-random` | 2500 | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q2_pool_random/round_1/model` | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q2_pool_random/round_1/training_round_summary.json` |
| 2 | `ns-error-mass` | 2500 | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q2_ns_error_mass/round_1/model` | `experiments/runs/imdb_qwen06b_T10_alpha010_train2500_lora/q2_ns_error_mass/round_1/training_round_summary.json` |





| ---: | --- | ---: | ---: | ---: |
| 1 | `experiments/eval_splits/imdb_q1_full_all_split_ids.json` | 49990 | 24999 | 24991 |
| 2 | `experiments/eval_splits/imdb_q2_full_all_split_ids.json` | 49990 | 36247 | 13743 |


| ---: | --- | ---: | ---: | ---: |
| 1 | `experiments/eval_splits/imdb_q1_base_wrong_all_split_ids.json` | 24996 | 24995 | 1 |
| 2 | `experiments/eval_splits/imdb_q2_base_wrong_all_split_ids.json` | 36246 | 36246 | 0 |

| --- | --- |
| base URL | `http://127.0.0.1:18021/v1` |
| api key | `EMPTY` |
| model path | `/teamspace/studios/this_studio/model/qwen3-0.6b` |
| round index | 1 |
| temperature | 0.0 |
| max tokens | 1 |
| top logprobs | 20 |
| parallel requests | 8192 |
| request retries | 3 |
| timeout | 180 |
| max model len | 4096 |
| max num seqs | 4096 |
| max num batched tokens | 1048576 |
| gpu memory utilization | 0.98 |
| enforce eager | true |


```bash
python scripts/predict_vllm_openai.py \
  --output_dir "$EVAL_DIR" \
  --round_index 1 \
  --model_path /teamspace/studios/this_studio/model/qwen3-0.6b \
  --data_path "experiments/inputs/imdb/query_id_${QID}/data.jsonl" \
  --split_ids_path "$EVAL_SPLIT_IDS" \
  --checkpoint_dir "$RUN_DIR/round_1/model" \
  --base_url http://127.0.0.1:18021/v1 \
  --api_key EMPTY \
  --temperature 0.0 \
  --max_tokens 1 \
  --top_logprobs 20 \
  --parallel_requests 8192 \
  --request_retries 3 \
  --timeout 180 \
  --max_model_len 4096 \
  --max_num_seqs 4096 \
  --max_num_batched_tokens 1048576 \
  --gpu_memory_utilization 0.98
```


- `experiments/evals/imdb_qwen06b_T10_alpha010_train2500_lora/q1_pool_random_full/`
- `experiments/evals/imdb_qwen06b_T10_alpha010_train2500_lora/q1_ns_error_mass_full/`
- `experiments/evals/imdb_qwen06b_T10_alpha010_train2500_lora/q2_pool_random_full/`
- `experiments/evals/imdb_qwen06b_T10_alpha010_train2500_lora/q2_ns_error_mass_full/`


- `experiments/reports/imdb_qwen06b_T10_alpha010_train2500_full_eval_summary.json`



| query_id | method | n | acc | macro-F1 | F1 no | F1 yes | balanced-acc | pred0 | pred1 | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `pool-random` | 49990 | 94.37% | 94.37% | 94.34% | 94.40% | 94.37% | 24,751 | 25,239 | 23,708/23,468/1,531/1,283 |
| 1 | `ns-error-mass` | 49990 | 92.27% | 92.25% | 92.60% | 91.91% | 92.27% | 27,202 | 22,788 | 21,957/24,168/831/3,034 |
| 2 | `pool-random` | 49990 | 88.69% | 85.72% | 92.24% | 79.20% | 85.47% | 36,557 | 13,433 | 10,762/33,576/2,671/2,981 |
| 2 | `ns-error-mass` | 49990 | 88.70% | 85.54% | 92.30% | 78.78% | 84.85% | 37,119 | 12,871 | 10,483/33,859/2,388/3,260 |



| query_id | method | train n | subset n | lora still correct | lora flips wrong | lora acc on subset | pred0 | pred1 | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `pool-random` | 2500 | 24,994 | 23,711 | 1,283 | 94.87% | 1,287 | 23,707 | 23,707/4/0/1,283 |
| 1 | `ns-error-mass` | 2500 | 24,994 | 21,960 | 3,034 | 87.86% | 3,038 | 21,956 | 21,956/4/0/3,034 |
| 2 | `pool-random` | 2500 | 13,744 | 10,763 | 2,981 | 78.31% | 2,982 | 10,762 | 10,762/1/0/2,981 |
| 2 | `ns-error-mass` | 2500 | 13,744 | 10,484 | 3,260 | 76.28% | 3,261 | 10,483 | 10,483/1/0/3,260 |


| query_id | method | train n | subset n | lora corrects base wrong | still wrong | lora acc on subset | pred0 | pred1 | TP/TN/FP/FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `pool-random` | 2500 | 24,996 | 23,465 | 1,531 | 93.88% | 23,464 | 1,532 | 1/23,464/1,531/0 |
| 1 | `ns-error-mass` | 2500 | 24,996 | 24,165 | 831 | 96.68% | 24,164 | 832 | 1/24,164/831/0 |
| 2 | `pool-random` | 2500 | 36,246 | 33,575 | 2,671 | 92.63% | 33,575 | 2,671 | 0/33,575/2,671/0 |
| 2 | `ns-error-mass` | 2500 | 36,246 | 33,858 | 2,388 | 93.41% | 33,858 | 2,388 | 0/33,858/2,388/0 |
