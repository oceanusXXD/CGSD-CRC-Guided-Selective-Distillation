

---





```
```



```
     ↓
     ↓
     ↓
     ↓
```

---



> **"Do We Really Need Difficulty-Aware Sampling? The Label Distribution Confound in Active LLM-to-SLM Distillation"**




---




---












---



2.1 Task and Protocol


$$\text{score}_i = \log p_\theta(\texttt{"1"} \mid x_i) - \log p_\theta(\texttt{"0"} \mid x_i), \quad \hat{y}_i = \mathbf{1}\{\text{score}_i > 0\}$$






$$\hat{\lambda} = \min\!\left\{\lambda \in [0.5, 1.0] : \frac{n_g}{n_g+1}\hat{R}(\lambda) + \frac{1}{n_g+1} \leq \alpha\right\}$$
$$D_\text{defer} = \{x : R_i < \hat{\lambda}\}, \quad D_\text{accept} = \mathcal{U}_\text{pool} \setminus D_\text{defer}$$

---



---




| Dataset / Query | True yes% | Base pred yes% | Base Macro-F1 | Prior Bias Type |
|----------------|-----------|----------------|---------------|-----------------|
| IMDb q1 (positive) | 50.0 | **99.99%** | 33.35 | Severe-yes |
| IMDb q2 (negative) | 27.5 | **100.0%** | 21.57 | Severe-yes |
| TwitterHate q1 | 83.2 | **100.0%** | 45.42 | Severe-yes |
| Codebase q1 (social) | 6.3 | **93.8%** | 12.30 | Severe-yes |
| Codebase q2 (CS) | 62.0 | **99.5%** | 39.60 | Severe-yes |
| Codebase q3 (factual) | 25.8 | **99.9%** | 20.65 | Severe-yes |
| FEVER (Qwen3-0.6B) | 52.4 | **10.0%** | 44.08 | Severe-no |
| FEVER (Qwen3-1.7B) | 52.4 | 58.9% | 86.72 | Near-balanced |

> "On 6 out of 7 classification queries, Qwen3-0.6B predicts one output label for over 93% of instances, achieving near-chance Macro-F1. We term such models *prior machines*: before fine-tuning, their output behavior is determined by prediction priors, not by input features."

> **Definition 1 (Prior Machine)**: A model $f_\theta$ is a *prior machine* on task $(q, \mathcal{U})$ if its zero-shot prediction distribution satisfies $|\Pr[\hat{y} = 1] - 0.5| > \delta$ for some $\delta > 0.3$, indicating strong prediction prior rather than discriminative classification.

---





$$\Pr[\hat{y}_i = c_\text{prior} \mid R_i \geq \hat{\lambda}] \gg \Pr[\hat{y}_i = c_\text{counter} \mid R_i \geq \hat{\lambda}]$$
$$\Pr[\hat{y}_i = c_\text{counter} \mid R_i < \hat{\lambda}] \gg \Pr[\hat{y}_i = c_\text{prior} \mid R_i < \hat{\lambda}]$$



| Dataset / Budget | Random: train yes% | Difficulty: train yes% | True yes% | Dist. Gap (Diff. vs. True) | Method |
|-----------------|-------------------|------------------------|-----------|---------------------------|--------|
| IMDb q1, n=2500 | 50.76% | 7.36% | 50.0% | **−42.6pp** | ns-error-mass |
| TwitterHate, n=50 | 78% | 66% | 83.2% | **−17.2pp** | crc-error-mass |
| FEVER 0.6B, n=1500 | 52.33% | 66.13% | 52.4% | **+13.7pp** | ns-error-mass |
| FEVER, n=500 | 52% | 66.6% | 52.4% | **+14.2pp** | defer-kcenter |
| FEVER 1.7B, n=2231 | 52.13% | 44.8% | 52.4% | **−7.6pp** | crc-error-mass |

> "Difficulty-aware methods consistently move training label distributions away from the true task distribution. The direction of shift matches the prior bias: for yes-biased models, hard samples are predominantly *no* labels (overcorrection); for no-biased models (FEVER), hard samples are predominantly *yes* labels (undercorrection)."

---






---



| Dataset | Budget | Method | Train yes% | True yes% | Dist. Gap | Pred yes% | Macro-F1 | vs. Random |
|---------|--------|--------|-----------|-----------|-----------|-----------|----------|-----------|
| TwitterHate | 50 | random | 78% | 83.2% | −5.2pp | 87.4% | 65.94 | — |
| TwitterHate | 50 | **crc-error-mass** | 66% | 83.2% | **−17.2pp** | 45.4% | **53.30** | **−12.64** |
| IMDb q1 | 2500 | random | 50.76% | 50% | +0.76pp | 50.5% | 94.37 | — |
| IMDb q1 | 2500 | **ns-error-mass** | 7.36% | 50% | **−42.6pp** | 45.6% | **92.25** | **−2.12** |
| FEVER | 500 | random-defer | 52% | 52.4% | −0.4pp | 71.8% | PF1=76.47 | — |
| FEVER | 500 | **defer-kcenter** | 66.6% | 52.4% | **+14.2pp** | 80.0% | PF1=**69.45** | **−7.0** |

> "In all overcorrection cases, the model's evaluation-set prediction yes-rate follows the training label distribution, not the true task distribution. This bidirectional effect — undercorrection in yes-biased settings and overcorrection in no-biased settings — is consistent with the prior machine hypothesis: the student learns the training label marginal, not the task."

---



| Condition | Sample Source | Train yes% | Eval pred yes% | Macro-F1 | vs. Balanced |
|-----------|---------------|-----------|----------------|----------|-------------|

>
> **Conclusion**: *Which* samples are selected (hard/easy, correct/wrong) is secondary. *How many samples of each label* are selected is primary.


---




| Model | Base pred yes% | Method | Train yes% | Macro-F1 | vs. Random | Dist. Gap |
|-------|---------------|--------|-----------|----------|-----------|-----------|
| 0.6B | 10% (severe-no) | random | 52.42% | 93.97 | — | 0pp |
| 0.6B | 10% (severe-no) | ns-error-mass | 68.2% | 94.47 | **+0.50** | +15.8pp |
| **1.7B** | **58.9% (near-balanced)** | random | 52.13% | 79.08 | — | 0pp |
| **1.7B** | **58.9% (near-balanced)** | crc-error-mass | 44.8% | **80.34** | **+1.26** | −7.6pp |
| **1.7B** | **58.9% (near-balanced)** | ns-difficulty-global | 67.98% | **79.88** | **+0.80** | +15.6pp |


$$\text{Use difficulty sampling if: } |\Pr_\text{base}[\hat{y}=1] - \hat{p}_1| < \delta_\text{threshold} \approx 10\text{pp}$$

---





$$\min_S \left|p_S^{(1)} - \hat{p}_1\right|, \quad \hat{p}_1 = \frac{1}{n_g}\sum_{j \in \mathcal{D}_\text{guide}} y_j$$

$$\max_S \sum_{i \in S} \text{Uncertainty}(x_i), \quad \text{Uncertainty} = 1 - R_i(T)$$

> **Remark 1**: Random sampling approximately satisfies Objective 1 (pool ≈ true distribution) but ignores Objective 2. Existing difficulty methods optimize Objective 2 while violating Objective 1. PCSS is the first method to explicitly satisfy both in order.

4.2 PCSS Algorithm（Algorithm 1 Box）

```
Algorithm 1: Prior-Corrective Stratified Selection (PCSS)
─────────────────────────────────────────────────────────
Input:  D_guide (1000 labeled), U_pool (unlabeled),
        Budget B, Risk α, Temperature T
Output: Training set S_train, Certified threshold λ*
─────────────────────────────────────────────────────────
Phase 1 — True Distribution Estimation:
  p̂₁ ← (1/n_g) Σ y_j  for j in D_guide

Phase 2 — CRC Calibration on D_guide:
  λ̂ ← argmin{λ : (n_g/(n_g+1))·R̂(λ) + 1/(n_g+1) ≤ α}
  D_defer ← {x ∈ U_pool : R(x,T) < λ̂}
  D_accept ← U_pool \ D_defer

Phase 3 — Stratified Budget Allocation:
  B₁ ← round(B · p̂₁)          // yes-label budget
  B₀ ← B - B₁                  // no-label budget

Phase 4 — Within-Stratum Difficulty Selection:
  // Within each stratum, prioritize defer (low R) over accept
  Ŷ ← student_predict(U_pool)  // proxy labels
  S₁ ← top-B₁ samples from D_defer∪D_accept where ŷᵢ=1, sorted by R(x,T)↑
  S₀ ← top-B₀ samples from D_defer∪D_accept where ŷᵢ=0, sorted by R(x,T)↑
  S_train ← teacher_label(S₁ ∪ S₀)  // LLM annotation

Phase 5 — Certification (on held-out D_cert):
  λ* ← CRC(D_cert, S_train, θ*, α)  // certified threshold

Return S_train, λ*
─────────────────────────────────────────────────────────
```

4.3 Properties




|------|------------|------------|------|



4.4 Current PCSS Evidence and What It Means



| Dataset | Budget | Comparable? | PCSS Macro-F1 | Random F1 | Difficulty F1 | PCSS vs. Random | PCSS vs. Difficulty | Property Checks |
|---------|--------|-------------|---------------|-----------|---------------|-----------------|---------------------|-----------------|
| TwitterHate | 250 | No | 47.33 | — | — | — | — | ✓/✓/✓ |
| TwitterHate | 2231 | Yes | 67.94 | 90.42 | 90.60 | −22.47 | −22.66 | ✓/✓/✓ |
| FEVER 0.6B | 1500 | Yes | 90.09 | 90.06 | 90.52 | +0.03 | −0.43 | ✓/✓/✓ |





$$
\Delta U(x\mid S)=w_u(1-r(x))+w_a\Delta\operatorname{Balance}(x\mid S)+w_d\Delta\operatorname{BucketCoverage}(x\mid S)
$$



---



5.1 The Certification Guarantee



$$\mathbb{E}_{\mathcal{D}_\text{cert},\,\mathcal{A}}\!\left[\mathbf{1}\{R(Z_\text{new}) \geq \hat{\lambda}^*\} \cdot \mathbf{1}\{\hat{y}(Z_\text{new}) \neq y(Z_\text{new})\}\right] \leq \alpha$$


> "This guarantee holds for PCSS, random sampling, and any strategy satisfying $\mathcal{D}_\text{cert} \perp \theta^*$. It is the non-trivial generalization bound to *unseen deployment samples*, not the trivial empirical risk bound on $\mathcal{D}_\text{cert}$ itself."

5.2 Practical Cost Analysis


| Component | LLM calls | SLM calls |
|-----------|-----------|-----------|
| Guide + Cert labeling | 1,200 | 0 |
| Full inference (Round 0) | 0 | $N$ |
| PCSS annotation | ~625 (with 25% buffer) | 0 |
| Deployment defer (<3%) | ~5,000 | 0 |
| **Total** | **~6,825** | **~1.6N** |


5.3 Why Guide-Set CRC Cannot Provide the Same Guarantee


> "The guide set $\mathcal{D}_\text{guide}$ participates in sampling decisions through diagnostic quantities ($e_\text{all}, e_\text{defer}, c_\text{crc}$), inducing dependence $\mathcal{D}_\text{guide} \not\perp \theta^*$. The exchangeability condition required by Lemma 1 (Angelopoulos et al., 2022) is violated; any calibration on $\mathcal{D}_\text{guide}$ provides only a heuristic estimate. Only the isolated certification set provides the formal guarantee."

---






> "Concurrent work reports diminishing returns of hard samples in reinforcement learning for SLMs [B1: Limits of Difficulty Scaling, 2604.06298], and step-length confounding in reasoning data selection [B2: Step Length Confounding, 2604.06834]. We identify a mechanistically distinct confound in supervised classification distillation: output prior bias creates structural coupling between sample difficulty and label distribution. Unlike [B1] (capacity boundary in RL) and [B2] (statistical artifact of step-length in reasoning), our mechanism is a design property of active selection under output prior bias."




---


6.1 The "Accidental Correctness" of Random Sampling


6.2 Practical Diagnostic Checklist

```
Step 1: Measure the round0 prediction yes-rate from existing prediction artifacts.
Step 2: Compare base pred-yes with true yes from guide/pool labels:
        If |pred-yes − true yes| ≥ 20pp or |pred-yes − 50%| ≥ 40pp:
             SEVERE risk; use PCSS/random before difficulty-only selection.
        If |pred-yes − true yes| ≥ 10pp or |pred-yes − 50%| ≥ 25pp:
             MODERATE risk; preserve distribution before using uncertainty.
        Otherwise:
             LOW risk; difficulty selection can be tested with CRC.
Step 3: After selection, verify train label distribution and proxy-label reliability.
Step 4: Certify deployment with an isolated certification set, not the guide set.
```

---



> "Before deploying any difficulty-aware selection strategy for SLM distillation, measure the model's output prior bias and verify that training label distribution is preserved — this single diagnostic step can prevent catastrophic performance failures."

---



---








---


|----|--------------------------------|---------|


---


|---------|------|---------|---------|

---







|-------------|------------|


---



---
