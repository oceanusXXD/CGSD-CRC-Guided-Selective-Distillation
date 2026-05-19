# FEVER 0.6B LoRA 平衡实验报告
## 页眉说明
- `balanced_seed1`：真实标签 yes/no 各 5000；base round0 在该集合上 acc=0.5340、F1=0.2189、pred_pos_rate=0.0966。
- `basef1_050`：真实标签 yes/no 各 5000，并构造成 base round0 F1=0.5000；base acc=0.6000、pred_pos_rate=0.3000。
- `basepred_label_quad_balanced_*`：四格均衡数据，按 `base_pred × label` 平衡构造。

## 修正 Prompt 后的 vLLM 复测结果

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

## 结果分析
- 500 规模 `lr1e-5_e2_r1_a16_all` 在两个测试集上都明显欠预测正类：`pred_pos_rate` 约 0.0005-0.0016，F1 接近 0。这说明修正 prompt 后，旧的“几乎全预测 1”结论被推翻，但 500 规模 e2 配置本身也不够好。
- 1000 规模 `lr1e-5_e2_r1_a16_all` 表现稳定提升：在 `balanced_seed1` 上 acc=0.7340/0.7482；在 `basef1_050` 上 acc=0.7065/0.7155，F1=0.7546/0.7663。
- 1000 规模 `lr1e-5_e2_r2_a16_all` 也有效：在 `balanced_seed1` 上 acc=0.7480/0.7483；在 `basef1_050` 上 acc=0.7025/0.7182。相比 r1，r2 的 full_random 在两个测试集上都不差，但 accept/defer 在 basef1_050 上略低于 r1。
- `lr1e-5_e3_r1_a16_all` 的两个 500 模型在 `basef1_050` 上也不错，acc 约 0.697-0.701，F1 约 0.739-0.746，说明 epoch=3 可以让 500 规模从欠预测正类中恢复，但还需要在 `balanced_seed1` 上补同配置验证。

## 1000 规模 epoch 扫描：basef1_050

### 固定参数

这一组只改变 `epoch` 和训练子集；其他训练/推理参数固定如下。

| 参数 | 值 |
|---|---|
| base model | `qwen3-0.6b` |
| test set | `balanced_test_10000_basef1_050_seed1` |
| train size | 1000 |
| lr | `1e-5` |
| LoRA | `r=1, alpha=16, target=attention_mlp, layer_scope=all` |
| max_length | 4096 |
| vLLM parallel_requests | 2000 |
| checkpoint policy | 训练到 e4，同时保存 `epoch_3` / `epoch_4` checkpoint |

### 变化参数：epoch = 3 vs 4

| train set | epoch | rows | complete | acc | F1 | TP | TN | FP | FN |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `full_random_balanced_1000_seed1` | 3 | 10000 | yes | 0.7175 | 0.7705 | 4741 | 2434 | 2566 | 259 |
| `full_random_balanced_1000_seed1` | 4 | 10000 | yes | 0.7237 | 0.7781 | 4845 | 2392 | 2608 | 155 |
| `accept15_defer85_random_balanced_1000_seed1` | 3 | 10000 | yes | 0.7188 | 0.7710 | 4734 | 2454 | 2546 | 266 |
| `accept15_defer85_random_balanced_1000_seed1` | 4 | 10000 | yes | 0.7141 | 0.7654 | 4665 | 2476 | 2524 | 335 |


### 小结：epoch 选择

| 对比项 | e3 | e4 | 当前判断 |
|---|---:|---:|---|
| `full_random_1000` F1 | 0.7705 | 0.7781 | e4 更好 |
| `accept15_defer85_random_1000` F1 | 0.7710 | 0.7654 | e3 更高 |
| 两个训练集平均 F1 | 0.7707 | 0.7718 | e4 略高 |

最终沿用 `epoch=4` 跑后续数据集。虽然 `accept15_defer85_random_1000` 单项 e3 更高，但两组平均 F1 上 e4 略高，且 `full_random_1000` 的 e4 提升更明显。

## 1000 规模 accept/defer vs full_random：e4 对比

### 固定参数

这一组固定训练和测试配置，只改变训练数据来源。对照基线是同规模的 `full_random_balanced_1000_seed1`。

| 参数 | 值 |
|---|---|
| test set | `balanced_test_10000_basef1_050_seed1` |
| train size | 1000 |
| epoch | 4 |
| lr | `1e-5` |
| LoRA | `r=1, alpha=16, target=attention_mlp, layer_scope=all` |
| vLLM parallel_requests | 2000 |
| full_random run root | `fever_balanced_lora_06b_lr1e5_e4_all_sweep1000_r1_alpha16` |
| accept/defer run root | `fever_balanced_lora_06b_lr1e5_e4_selected_sweep1000_r1_alpha16` |

### 对照基线：full_random_1000

| train set | sampling | accept/defer split | acc | F1 | pred_pos_rate | precision | recall |
|---|---|---|---:|---:|---:|---:|---:|
| `full_random_balanced_1000_seed1` | random from full pool | n/a | 0.7237 | 0.7781 | 0.7453 | 0.6501 | 0.9690 |

### 总览：accept/defer 相对 full_random 的变化

`Δ` 列均为相对 `full_random_balanced_1000_seed1` e4 的差值。

| train set | sampling | accept/defer split | acc | Δacc | F1 | ΔF1 | recall | Δrecall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `accept15_defer85_random_balanced_1000_seed1` | random | 15/85 | 0.7141 | -0.0096 | 0.7654 | -0.0127 | 0.9330 | -0.0360 |
| `accept30_defer70_random_balanced_1000_seed1` | random | 30/70 | 0.7026 | -0.0211 | 0.7413 | -0.0368 | 0.8524 | -0.1166 |
| `accept20_defer80_random_balanced_1000_seed1` | random | 20/80 | 0.6769 | -0.0468 | 0.6992 | -0.0789 | 0.7510 | -0.2180 |
| `accept15_defer85_kcenter_balanced_1000_seed1` | kcenter | 15/85 | 0.6846 | -0.0391 | 0.7076 | -0.0705 | 0.7632 | -0.2058 |
| `accept30_defer70_kcenter_balanced_1000_seed1` | kcenter | 30/70 | 0.6546 | -0.0691 | 0.6580 | -0.1201 | 0.6644 | -0.3046 |

### 只看 random 采样：accept/defer 比例变化

| train set | accept/defer split | acc | F1 | pred_pos_rate | precision | recall | 相对 full_random |
|---|---|---:|---:|---:|---:|---:|---|
| `full_random_balanced_1000_seed1` | n/a | 0.7237 | 0.7781 | 0.7453 | 0.6501 | 0.9690 | baseline |
| `accept15_defer85_random_balanced_1000_seed1` | 15/85 | 0.7141 | 0.7654 | 0.7189 | 0.6489 | 0.9330 | 最接近 full_random，F1 -0.0127 |
| `accept30_defer70_random_balanced_1000_seed1` | 30/70 | 0.7026 | 0.7413 | 0.6498 | 0.6559 | 0.8524 | precision 略高，但 recall 明显低 |
| `accept20_defer80_random_balanced_1000_seed1` | 20/80 | 0.6769 | 0.6992 | 0.5741 | 0.6541 | 0.7510 | 正类预测不足，F1 -0.0789 |

### 只看 kcenter 采样：accept/defer 比例变化

| train set | accept/defer split | acc | F1 | pred_pos_rate | precision | recall | 相对 full_random |
|---|---|---:|---:|---:|---:|---:|---|
| `full_random_balanced_1000_seed1` | n/a | 0.7237 | 0.7781 | 0.7453 | 0.6501 | 0.9690 | baseline |
| `accept15_defer85_kcenter_balanced_1000_seed1` | 15/85 | 0.6846 | 0.7076 | 0.5786 | 0.6595 | 0.7632 | kcenter 中较好，但 F1 -0.0705 |
| `accept30_defer70_kcenter_balanced_1000_seed1` | 30/70 | 0.6546 | 0.6580 | 0.5098 | 0.6516 | 0.6644 | 最弱，recall 掉得最多 |

### 混合构造：global random 100 + accept random 150 + defer kcenter 750

这组固定 `epoch=4`、`lr=1e-5`、1000 条训练数据，只改变训练集构成。混合集命名为 `global_random100_accept_random150_defer_kcenter750_balanced_1000_seed1`，标签仍保持 500/500。

| train set | 训练构成 | acc | Δacc | F1 | ΔF1 | pred_pos_rate | precision | recall | Δrecall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_random_balanced_1000_seed1` | full random 1000 | 0.7237 | baseline | 0.7781 | baseline | 0.7453 | 0.6501 | 0.9690 | baseline |
| `global_random100_accept_random150_defer_kcenter750_balanced_1000_seed1` | full random 100 + accept random 150 + defer kcenter 750 | 0.6576 | -0.0661 | 0.6553 | -0.1228 | 0.4934 | 0.6597 | 0.6510 | -0.3180 |

这个混合构造比 full_random 更保守：precision 略高，但 `pred_pos_rate` 从 0.7453 降到 0.4934，recall 大幅下降，导致 F1 明显低于 full_random。

### 小结：accept/defer 与 full_random 的差异

- `full_random_balanced_1000_seed1` e4 仍是这批里最强：F1=0.7781、acc=0.7237。
- accept/defer 构造普遍降低 recall；precision 基本持平或略高，但不足以抵消 recall 下降。
- random accept/defer 明显强于 kcenter accept/defer。最佳 accept/defer 是 `accept15_defer85_random_balanced_1000_seed1`，与 full_random 最接近，F1 低 0.0127。
- 四个新增完整跑完的数据集中，最佳是 `accept30_defer70_random_balanced_1000_seed1`，F1=0.7413，比 full_random 低 0.0368。
- 新测的 `global_random100_accept_random150_defer_kcenter750_balanced_1000_seed1` 不建议作为后续默认方案：F1=0.6553，比 full_random 低 0.1228，主要问题是 recall 过低。
