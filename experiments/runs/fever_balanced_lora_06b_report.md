

| test set | train size | train set | config | acc | F1 | pred_pos_rate | precision | recall |
|---|---:|---|---|---:|---:|---:|---:|---:|
| balanced_seed1 | 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_all | 0.5007 | 0.0036 | 0.0011 | 0.8182 | 0.0018 |
| balanced_seed1 | 500 | full_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_all | 0.5008 | 0.0048 | 0.0016 | 0.7500 | 0.0024 |
| balanced_seed1 | 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r1_a16_all | 0.7340 | 0.7619 | 0.6172 | 0.6896 | 0.8512 |
| balanced_seed1 | 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r1_a16_all | 0.7482 | 0.7843 | 0.6674 | 0.6859 | 0.9156 |
| balanced_seed1 | 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r2_a16_all | 0.7480 | 0.7833 | 0.6630 | 0.6870 | 0.9110 |
| balanced_seed1 | 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r2_a16_all | 0.7483 | 0.7831 | 0.6603 | 0.6880 | 0.9086 |
| basef1_050 | 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_all | 0.5001 | 0.0012 | 0.0005 | 0.6000 | 0.0006 |
| basef1_050 | 500 | full_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_all | 0.5000 | 0.0012 | 0.0006 | 0.5000 | 0.0006 |
| basef1_050 | 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r2_a16_all | 0.7025 | 0.7477 | 0.6793 | 0.6491 | 0.8818 |
| basef1_050 | 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r2_a16_all | 0.7182 | 0.7713 | 0.7324 | 0.6490 | 0.9506 |
| basef1_050 | 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r1_a16_all | 0.7065 | 0.7546 | 0.6961 | 0.6483 | 0.9026 |
| basef1_050 | 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r1_a16_all | 0.7155 | 0.7663 | 0.7175 | 0.6502 | 0.9330 |
| basef1_050 | 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e3_r1_a16_all | 0.7009 | 0.7457 | 0.6761 | 0.6486 | 0.8770 |
| basef1_050 | 500 | full_random_balanced_500_seed1 | lr1e-5_e3_r1_a16_all | 0.6968 | 0.7390 | 0.6616 | 0.6487 | 0.8584 |





|---|---|
| base model | `qwen3-0.6b` |
| test set | `balanced_test_10000_basef1_050_seed1` |
| train size | 1000 |
| lr | `1e-5` |
| LoRA | `r=1, alpha=16, target=attention_mlp, layer_scope=all` |
| max_length | 4096 |
| vLLM parallel_requests | 2000 |


| train set | epoch | rows | complete | acc | F1 | TP | TN | FP | FN |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `full_random_balanced_1000_seed1` | 3 | 10000 | yes | 0.7175 | 0.7705 | 4741 | 2434 | 2566 | 259 |
| `full_random_balanced_1000_seed1` | 4 | 10000 | yes | 0.7237 | 0.7781 | 4845 | 2392 | 2608 | 155 |
| `accept15_defer85_random_balanced_1000_seed1` | 3 | 10000 | yes | 0.7188 | 0.7710 | 4734 | 2454 | 2546 | 266 |
| `accept15_defer85_random_balanced_1000_seed1` | 4 | 10000 | yes | 0.7141 | 0.7654 | 4665 | 2476 | 2524 | 335 |



|---|---:|---:|---|





|---|---|
| test set | `balanced_test_10000_basef1_050_seed1` |
| train size | 1000 |
| epoch | 4 |
| lr | `1e-5` |
| LoRA | `r=1, alpha=16, target=attention_mlp, layer_scope=all` |
| vLLM parallel_requests | 2000 |
| full_random run root | `fever_balanced_lora_06b_lr1e5_e4_all_sweep1000_r1_alpha16` |
| accept/defer run root | `fever_balanced_lora_06b_lr1e5_e4_selected_sweep1000_r1_alpha16` |


| train set | sampling | accept/defer split | acc | F1 | pred_pos_rate | precision | recall |
|---|---|---|---:|---:|---:|---:|---:|
| `full_random_balanced_1000_seed1` | random from full pool | n/a | 0.7237 | 0.7781 | 0.7453 | 0.6501 | 0.9690 |



| train set | sampling | accept/defer split | acc | Δacc | F1 | ΔF1 | recall | Δrecall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `accept15_defer85_random_balanced_1000_seed1` | random | 15/85 | 0.7141 | -0.0096 | 0.7654 | -0.0127 | 0.9330 | -0.0360 |
| `accept30_defer70_random_balanced_1000_seed1` | random | 30/70 | 0.7026 | -0.0211 | 0.7413 | -0.0368 | 0.8524 | -0.1166 |
| `accept20_defer80_random_balanced_1000_seed1` | random | 20/80 | 0.6769 | -0.0468 | 0.6992 | -0.0789 | 0.7510 | -0.2180 |
| `accept15_defer85_kcenter_balanced_1000_seed1` | kcenter | 15/85 | 0.6846 | -0.0391 | 0.7076 | -0.0705 | 0.7632 | -0.2058 |
| `accept30_defer70_kcenter_balanced_1000_seed1` | kcenter | 30/70 | 0.6546 | -0.0691 | 0.6580 | -0.1201 | 0.6644 | -0.3046 |


|---|---|---:|---:|---:|---:|---:|---|
| `full_random_balanced_1000_seed1` | n/a | 0.7237 | 0.7781 | 0.7453 | 0.6501 | 0.9690 | baseline |


|---|---|---:|---:|---:|---:|---:|---|
| `full_random_balanced_1000_seed1` | n/a | 0.7237 | 0.7781 | 0.7453 | 0.6501 | 0.9690 | baseline |



|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_random_balanced_1000_seed1` | full random 1000 | 0.7237 | baseline | 0.7781 | baseline | 0.7453 | 0.6501 | 0.9690 | baseline |
| `global_random100_accept_random150_defer_kcenter750_balanced_1000_seed1` | full random 100 + accept random 150 + defer kcenter 750 | 0.6576 | -0.0661 | 0.6553 | -0.1228 | 0.4934 | 0.6597 | 0.6510 | -0.3180 |
