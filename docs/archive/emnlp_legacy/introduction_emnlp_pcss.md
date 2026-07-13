



```text
```




```json
```




```text
score = logit_or_logprob("1") - logit_or_logprob("0")
```









| --- | --- | --- |




```text

Query: ...
Document: ...
```




| --- | --- | --- |






```text
routing_score = sigmoid(abs(score) / temperature)
```




```text
lambda_grid = 0.50, 0.51, 0.52, ..., 1.00
```


```text
accept = routing_score >= lambda
defer = routing_score < lambda
```


```text
empirical_risk = wrong_accept_count / guide_count
risk_bound = guide_count / (guide_count + 1) * empirical_risk + 1 / (guide_count + 1)
```



```text
risk_bound <= alpha
```



| --- | --- | --- |


- `routing_score`
- `routing_temperature`
- `crc_decision`
- `defer`
- `decision_threshold`
- `tau_crc`


```text
tau_crc = temperature * log(lambda_hat / (1 - lambda_hat))
```




```text
Input:
  method                   # random / pcss / crc-error-mass

  for row in guide_predictions:
      row.routing_score = sigmoid(abs(row.score) / temperature)

  lambda_hat = 1.01
  for lambda in [0.50, 0.51, ..., 1.00]:
      accept_rows = []
      wrong_accept_count = 0

      for row in guide_predictions:
          if row.routing_score >= lambda:
              accept_rows.append(row)
              if row.prediction != row.label:
                  wrong_accept_count += 1

      empirical_risk = wrong_accept_count / len(guide_predictions)
      risk_bound = len(guide_predictions) / (len(guide_predictions) + 1) * empirical_risk
                 + 1 / (len(guide_predictions) + 1)

      if risk_bound <= alpha:
          lambda_hat = lambda
          break

  routed_guide = []
  for row in guide_predictions:
      if lambda_hat > 1.0:
          row.decision = "defer"
      else if row.routing_score >= lambda_hat:
          row.decision = "accept"
      else:
          row.decision = "defer"
      routed_guide.append(row)

  routed_pool = []
  for row in pool_predictions:
      row.routing_score = sigmoid(abs(row.score) / temperature)

      if lambda_hat > 1.0:
          row.decision = "defer"
      else if row.routing_score >= lambda_hat:
          row.decision = "accept"
      else:
          row.decision = "defer"
      routed_pool.append(row)

  candidate_pool = []
  for row in routed_pool:
      if row.id not in blocked_ids:
          candidate_pool.append(row)

  if method == "random":
      selected = RANDOM_SELECT(candidate_pool, budget, seed)

  if method == "pcss":
      selected = PCSS_SELECT(
          routed_guide,
          candidate_pool,
          budget,
          temperature
      )

  if method == "crc-error-mass":
      selected = CRC_ERROR_MASS_SELECT(
          routed_guide,
          candidate_pool,
          budget,
          seed,
          accept_strategy,
          defer_strategy
      )

Output:
```


```text
function RANDOM_SELECT(candidate_pool, budget, seed):
    ids = unique ids from candidate_pool
    shuffle(ids, seed)
    return first min(budget, len(ids)) rows
```


```text
function PCSS_SELECT(guide_rows, pool_rows, budget, temperature):
    guide_label1_count = count(row.label == 1 for row in guide_rows)
    p_hat_1 = guide_label1_count / len(guide_rows)

    target_label1_budget = round_half_up(budget * p_hat_1)
    target_label0_budget = budget - target_label1_budget

    label1_candidates = [row for row in pool_rows if row.prediction == 1]
    label0_candidates = [row for row in pool_rows if row.prediction == 0]

    B_label1 = min(target_label1_budget, len(label1_candidates))
    B_label0 = min(target_label0_budget, len(label0_candidates))
    remaining = min(budget, len(pool_rows)) - B_label1 - B_label0

    if remaining > 0:
        add_to_label0 = min(remaining, len(label0_candidates) - B_label0)
        B_label0 += add_to_label0
        remaining -= add_to_label0

    if remaining > 0:
        add_to_label1 = min(remaining, len(label1_candidates) - B_label1)
        B_label1 += add_to_label1

    sort label0_candidates by (routing_score ascending, id ascending)
    sort label1_candidates by (routing_score ascending, id ascending)

    selected_label0 = first B_label0 rows from label0_candidates
    selected_label1 = first B_label1 rows from label1_candidates

    return selected_label1 + selected_label0
```


```text
function CRC_ERROR_MASS_SELECT(
    guide_rows,
    pool_rows,
    budget,
    seed,
    accept_strategy,
    defer_strategy
):
    guide_defer_rows = [row for row in guide_rows if row.decision == "defer"]
    pool_defer_rows = [row for row in pool_rows if row.decision == "defer"]
    pool_accept_rows = [row for row in pool_rows if row.decision == "accept"]

    r_U = len(pool_defer_rows) / len(pool_rows)
    r_C = len(guide_defer_rows) / len(guide_rows)

    guide_error_count = count(row.prediction != row.label for row in guide_rows)
    guide_defer_error_count = count(row.prediction != row.label for row in guide_defer_rows)

    e_all = guide_error_count / len(guide_rows)

    if len(guide_defer_rows) == 0 or guide_error_count == 0 or e_all <= 0:
        e_defer = 0.0
        c_crc = 1.0
        eta_crc = 0.0
    else:
        e_defer = guide_defer_error_count / len(guide_defer_rows)
        c_crc = e_defer / e_all

        if c_crc <= 1.0 or r_C <= 0.0 or r_C >= 1.0:
            eta_crc = 0.0
        else:
            eta_crc = clamp(log(c_crc) / log(1 / r_C), 0.0, 1.0)

    s_defer = clamp(r_U + eta_crc * (1 - r_U)^2, 0.0, 1.0)
    s_accept = 1.0 - s_defer

    B_defer = round_half_up(budget * s_defer)
    B_accept = budget - B_defer

    B_defer = min(B_defer, len(pool_defer_rows))
    B_accept = min(B_accept, len(pool_accept_rows))

    selected_accept = SELECT_SIDE(pool_accept_rows, B_accept, seed, accept_strategy)
    selected_defer = SELECT_SIDE(pool_defer_rows, B_defer, seed + 1, defer_strategy)

    return selected_accept + selected_defer

function SELECT_SIDE(rows, k, seed, strategy):
    if strategy == "random":
        shuffle(rows, seed)
        return first k rows

    if strategy == "high-confidence":
        sort rows by (routing_score descending, id ascending)
        return first k rows
```

6.1 random




| --- | --- | --- |





```text
p_hat_1 = guide_label1_count / guide_count
```


```text
target_label1_budget = round_half_up(budget * p_hat_1)
target_label0_budget = budget - target_label1_budget
```


```text
proxy_label = prediction
```


```text
```





| --- | --- | --- |




```text
r_U = pool_defer_count / pool_total
r_C = guide_defer_count / guide_count
e_all = guide_error_count / guide_count
e_defer = guide_defer_error_count / guide_defer_count
```




```text
c_crc = e_defer / e_all
```


```text
eta_crc = log(c_crc) / log(1 / r_C)
```



```text
s_defer = r_U + eta_crc * (1 - r_U)^2
s_accept = 1 - s_defer
```


```text
B_defer = round_half_up(budget * s_defer)
B_accept = budget - B_defer
```







| --- | --- | --- |



- `teacher_label`
- `teacher_confidence`
- `teacher_logit_margin`






```text
```





| --- | --- |
| `qv` | `q_proj`, `v_proj` |
| `qkvo` / `attention` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `mlp` | `gate_proj`, `up_proj`, `down_proj` |


| --- | --- |


| --- | --- | --- |
| `lora_r` | `1` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `lora_dropout` | `0.05` | LoRA dropout |



```text
class_weight[label] = total_count / (2 * count(label))
```





```text
score = logit("1") - logit("0")
```



| --- | --- | --- |
| `batch_size` | `64` | batch size |







```text
Query:
{query}

Document:
{document}
```





| --- | --- | --- |
| `tensor_parallel_size` | `1` | vLLM tensor parallel |


- `embeddings.npy`
- `embeddings.ids.jsonl`
- `embeddings.meta.json`




| --- | --- |




```text
split_ids.json
prepare_usage.json

round_0/
  all_student_predictions.jsonl
  guide_student_predictions.jsonl
  final_student_predictions.jsonl
  pool_student_predictions.jsonl
  guide_crc_predictions.jsonl
  pool_crc_predictions.jsonl
  crc_summary.json
  selected_train_rows.jsonl
  selection_summary.json

train_rows.jsonl

round_1/
  model/
    adapter/
    model_config.json
  training_rows_used.jsonl
  train_label_snapshot.json
  training_round_summary.json
```
