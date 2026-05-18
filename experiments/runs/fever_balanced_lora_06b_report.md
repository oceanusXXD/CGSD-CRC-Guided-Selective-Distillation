# FEVER 0.6B LoRA 平衡实验报告

## 页眉说明
- `balanced_seed1`：第一个 balanced 测试集，只约束真实标签均衡，即 yes/no（label 1/0）各 5000；没有约束 base 预测分布。base 在该集合上 acc=0.5340、F1=0.2189、预测正类比例=0.0966。
- `basef1_050`：第二个测试集，同样约束真实标签 yes/no 各 5000，同时构造成 base round0 在该集合上 F1=0.5000。base 在该集合上 acc=0.6000、预测正类比例=0.3000。
- `basepred_label_quad_balanced_1000_seed1`：四格均衡训练集，按 `base_pred × label` 构造，`(base_pred=0,label=0)`、`(base_pred=0,label=1)`、`(base_pred=1,label=0)`、`(base_pred=1,label=1)` 各 250。
- `basepred_label_quad_balanced_500_seed1`：从 full pool 构造的四格均衡 500 训练集，每格 125；半 epoch 版本实际训练 250 条，四格约为 62/62/62/64。
- `accept15_defer85_basepred_label_quad_balanced_500_seed1`：accept15/defer85 来源的 500 训练集，不是严格四格均衡；四格为 38/213/212/37，accept/defer 为 75/425；半 epoch 版本训练 250 条。
- 表格不是完整超参 sweep，只汇总当前已完成并用于对照的 base、500/1000 LoRA、四格数据实验，以及最新低学习率实验。

## 结果分析
- 当前结论更新后仍然一致：LoRA 的 F1 高主要来自预测正类比例过高，而不是整体判别能力超过 base。大多数 LoRA 的预测正类比例在 0.95 到 0.99 附近，recall 很高，但 precision 基本只有 0.49 到 0.50。
- 在 `balanced_seed1` 上，base acc=0.5340、F1=0.2189、预测正类比例=0.0966；多数 LoRA 的 F1 到约 0.65，但 acc 基本在 0.49 左右，说明模型接近“几乎全预测 1”。
- 在 `basef1_050` 上，base acc=0.6000、F1=0.5000、预测正类比例=0.3000；已测 LoRA 的 acc 仍约 0.49 到 0.50，低于 base。
- 四格均衡训练集是更合理的数据方向，但当前已测配置下仍未解决正类塌缩。`lr5e-6_e1_r1_a8_all` 在四格 1000 上的预测正类比例仍为 0.9511/0.9542；四格 500 半 epoch 也仍为 0.9690 左右。
- 最新把学习率降到极低后，结论没有改善：`lr1e-8_e1_r1_a4` 在 `basef1_050` 上 acc=0.4976、F1=0.6625、预测正类比例=0.9886；`lr3e-7_e1_r1_a4_all` 在 `basef1_050` 上 acc=0.4976、F1=0.6625、预测正类比例=0.9888。这说明问题不只是学习率过大；更可能是训练目标/LoRA 更新方向把第一个输出 token 的分布整体推向 `1`。
- 已有预测文件里的 `vllm_raw_text` 与 `logprob(1)-logprob(0)` 判定一致，说明不是后处理阈值把本来生成的 `0` 改成了 `1`；模型实际首 token 就大量生成 `1`。
- 下一步定位应优先做三个对照：base 原始模型、未训练 LoRA adapter、训练后 LoRA，三者都用直接生成 1 token 的 `vllm_raw_text` 分布对比。如果未训练 LoRA 正常、训练后塌缩，则问题在训练更新；如果未训练 LoRA 就塌缩，则要查 adapter 包装/加载/推理路径。

## 数据检查
- 1000 四格训练集：rows=1000，四格分布为 {'pred0_label0': 250, 'pred0_label1': 250, 'pred1_label0': 250, 'pred1_label1': 250}。
- 500 full-pool 四格训练集：rows=500，四格分布为 {'pred0_label0': 125, 'pred0_label1': 125, 'pred1_label0': 125, 'pred1_label1': 125}；halfepoch rows=250，四格为 {'pred0_label0': 62, 'pred0_label1': 62, 'pred1_label0': 62, 'pred1_label1': 64}。
- 500 accept15/defer85 训练集：rows=500，四格为 {'pred0_label0': 38, 'pred0_label1': 213, 'pred1_label0': 212, 'pred1_label1': 37}，accept/defer={'accept': 75, 'defer': 425}；halfepoch rows=250，四格为 {'pred0_label0': 19, 'pred0_label1': 107, 'pred1_label0': 106, 'pred1_label1': 18}，accept/defer={'accept': 37, 'defer': 213}。
- Base-F1 测试集 `balanced_test_10000_basef1_050_seed1`：acc=0.6000，F1=0.5000，label 0/1=5000/5000。

## 结果：balanced_seed1
| 规模 | 训练集 | 配置 | loss | acc | F1 | 预测正类比例 | precision | recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | base | base qwen3-0.6b |  | 0.5340 | 0.2189 | 0.0966 | 0.6760 | 0.1306 |
| 500 | accept15_defer85_basepred_label_quad_balanced_500_seed1 | lr5e-6_halfepoch_r1_a8_all_quad500 | 21.1443 | 0.4954 | 0.6597 | 0.9826 | 0.4977 | 0.9780 |
| 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_all | 2.1847 | 0.4879 | 0.6524 | 0.9731 | 0.4938 | 0.9610 |
| 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_last1 | 20.4288 | 0.4957 | 0.6584 | 0.9761 | 0.4978 | 0.9718 |
| 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e2_r1_a8_all | 11.0497 | 0.4913 | 0.6552 | 0.9753 | 0.4955 | 0.9666 |
| 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e3_r1_a16_all | 0.3283 | 0.4812 | 0.6414 | 0.9466 | 0.4901 | 0.9278 |
| 500 | basepred_label_quad_balanced_500_seed1 | lr5e-6_halfepoch_r1_a8_all_quad500 | 20.9705 | 0.4942 | 0.6557 | 0.9690 | 0.4970 | 0.9632 |
| 500 | full_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_all | 2.1345 | 0.4882 | 0.6524 | 0.9724 | 0.4939 | 0.9606 |
| 500 | full_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_last1 | 20.4144 | 0.4962 | 0.6603 | 0.9832 | 0.4981 | 0.9794 |
| 500 | full_random_balanced_500_seed1 | lr1e-5_e2_r1_a8_all | 10.9509 | 0.4899 | 0.6528 | 0.9691 | 0.4948 | 0.9590 |
| 500 | full_random_balanced_500_seed1 | lr1e-5_e3_r1_a16_all | 0.2880 | 0.4856 | 0.6483 | 0.9626 | 0.4925 | 0.9482 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r1_a16_all | 0.3079 | 0.4926 | 0.6573 | 0.9808 | 0.4962 | 0.9734 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r1_a8_all | 0.5730 | 0.4858 | 0.6485 | 0.9630 | 0.4926 | 0.9488 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r2_a16_all | 0.3048 | 0.4905 | 0.6554 | 0.9787 | 0.4951 | 0.9692 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r2_a8_all | 0.9566 | 0.4867 | 0.6501 | 0.9669 | 0.4931 | 0.9536 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr5e-6_e1_r1_a8_all | 18.8541 | 0.4923 | 0.6536 | 0.9655 | 0.4960 | 0.9578 |
| 1000 | basepred_label_quad_balanced_1000_seed1 | lr5e-6_e1_r1_a8_all | 18.9941 | 0.4869 | 0.6464 | 0.9511 | 0.4931 | 0.9380 |
| 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r1_a16_all | 0.2522 | 0.4908 | 0.6553 | 0.9772 | 0.4953 | 0.9680 |
| 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r1_a8_all | 0.5237 | 0.4896 | 0.6538 | 0.9742 | 0.4947 | 0.9638 |
| 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r2_a16_all | 0.2605 | 0.4919 | 0.6567 | 0.9799 | 0.4959 | 0.9718 |
| 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r2_a8_all | 0.9319 | 0.4889 | 0.6521 | 0.9693 | 0.4943 | 0.9582 |

## 结果：basef1_050
| 规模 | 训练集 | 配置 | loss | acc | F1 | 预测正类比例 | precision | recall |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | base | base qwen3-0.6b |  | 0.6000 | 0.5000 | 0.3000 | 0.6667 | 0.4000 |
| 500 | accept15_defer85_basepred_label_quad_balanced_500_seed1 | lr5e-6_halfepoch_r1_a8_all_quad500 | 21.1443 | 0.4971 | 0.6603 | 0.9805 | 0.4985 | 0.9776 |
| 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_all | 2.1847 | 0.4927 | 0.6557 | 0.9733 | 0.4962 | 0.9660 |
| 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_last1 | 20.4288 | 0.4971 | 0.6619 | 0.9873 | 0.4985 | 0.9844 |
| 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e2_r1_a8_all | 11.0497 | 0.4913 | 0.6514 | 0.9593 | 0.4955 | 0.9506 |
| 500 | accept15_defer85_random_balanced_500_seed1 | lr1e-5_e3_r1_a16_all | 0.3283 | 0.4865 | 0.6465 | 0.9527 | 0.4929 | 0.9392 |
| 500 | basepred_label_quad_balanced_500_seed1 | lr5e-6_halfepoch_r1_a8_all_quad500 | 20.9705 | 0.4982 | 0.6584 | 0.9690 | 0.4991 | 0.9672 |
| 500 | full_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_all | 2.1345 | 0.4934 | 0.6571 | 0.9774 | 0.4966 | 0.9708 |
| 500 | full_random_balanced_500_seed1 | lr1e-5_e2_r1_a16_last1 | 20.4144 | 0.4972 | 0.6606 | 0.9814 | 0.4986 | 0.9786 |
| 500 | full_random_balanced_500_seed1 | lr1e-5_e2_r1_a8_all | 10.9509 | 0.4942 | 0.6578 | 0.9780 | 0.4970 | 0.9722 |
| 500 | full_random_balanced_500_seed1 | lr1e-5_e3_r1_a16_all | 0.2880 | 0.4909 | 0.6525 | 0.9649 | 0.4953 | 0.9558 |
| 500 | full_random_balanced_500_seed1 | lr1e-8_e1_r1_a4_all | 20.9700 | 0.4976 | 0.6625 | 0.9886 | 0.4988 | 0.9862 |
| 500 | full_random_balanced_500_seed1 | lr3e-7_e1_r1_a4_all | 20.9575 | 0.4976 | 0.6625 | 0.9888 | 0.4988 | 0.9864 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r1_a16_all | 0.3079 | 0.4949 | 0.6584 | 0.9785 | 0.4974 | 0.9734 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r1_a8_all | 0.5730 | 0.4914 | 0.6525 | 0.9634 | 0.4955 | 0.9548 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r2_a16_all | 0.3048 | 0.4964 | 0.6609 | 0.9852 | 0.4982 | 0.9816 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr1e-5_e2_r2_a8_all | 0.9566 | 0.4870 | 0.6465 | 0.9512 | 0.4932 | 0.9382 |
| 1000 | accept15_defer85_random_balanced_1000_seed1 | lr5e-6_e1_r1_a8_all | 18.8541 | 0.4938 | 0.6510 | 0.9506 | 0.4967 | 0.9444 |
| 1000 | basepred_label_quad_balanced_1000_seed1 | lr5e-6_e1_r1_a8_all | 18.9941 | 0.4910 | 0.6500 | 0.9542 | 0.4953 | 0.9452 |
| 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r1_a16_all | 0.2522 | 0.4948 | 0.6577 | 0.9758 | 0.4973 | 0.9706 |
| 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r1_a8_all | 0.5237 | 0.4944 | 0.6575 | 0.9760 | 0.4971 | 0.9704 |
| 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r2_a16_all | 0.2605 | 0.4952 | 0.6588 | 0.9796 | 0.4976 | 0.9748 |
| 1000 | full_random_balanced_1000_seed1 | lr1e-5_e2_r2_a8_all | 0.9319 | 0.4928 | 0.6549 | 0.9696 | 0.4963 | 0.9624 |

## 结果文件
- CSV: `experiments/runs/fever_balanced_lora_06b_final_table.csv`
- JSON: `experiments/runs/fever_balanced_lora_06b_final_table.json`
