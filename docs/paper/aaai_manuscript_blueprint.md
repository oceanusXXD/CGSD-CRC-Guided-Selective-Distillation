# MIAS/DCMS AAAI Manuscript Blueprint

> Status: a writing blueprint, not a result report. Replace every bracketed
> placeholder only with a frozen run record or paper artifact.
>
> Canonical terms: **MIAS** (Model-Induced Acquisition Shift) and **DCMS**
> (Distribution-Constrained Model-Guided Selection). Do not introduce SISS,
> DGA, PCSS, CRC, or AED as competing main-method names in the paper.

## 1. Proposed Paper Claim

**One-sentence contribution to confirm before drafting:**

> Model-dependent acquisition can induce group-specific supervision shift; MIAS
> measures and tests this transmission, while DCMS constrains observable
> coverage without discarding the base selector's utility signal.

The paper is not about proving that uncertainty sampling is always harmful, that
matching the pool distribution is universally optimal, or that DCMS eliminates
model bias. The paper asks when the state of a selector changes who receives
supervision, whether that change reaches downstream group behavior, and whether
DCMS improves the resulting utility-coverage trade-off.

## 2. Evidence Contract

| Intended claim | Required evidence | Current paper treatment |
|---|---|---|
| Selector scores depend on task strata | Fixed pool, score-stratum statistic, Random-calibrated acquisition shift | AG News diagnostic evidence only; no cross-task claim yet |
| Selector state causes acquisition shift | Class-intercept or length-coefficient response curve with zero-point reproduction | Planned; no causal language before completion |
| Supervision composition changes downstream behavior | Matched-utility or composition intervention plus group metrics | Planned |
| DCMS improves a utility-coverage trade-off | Paired base-selector vs `+DCMS`, equal budgets/tokens, seeds, CIs | Planned |
| Preference selection extends beyond class labels | Reliable dual-order scorer, attribute audits, held-out preference/capability evaluation | Current CPU pilot validates execution only |
| Binary evidence generalizes | New native-label datasets, frozen splits, sample-level records, reruns | Data is ready; no result claim yet |

Only the rows whose required evidence is complete become result paragraphs in
the main paper. The remaining rows become experiment plans, limitations, or are
removed before submission.

## 3. Narrative and Title

### Narrative

1. A current model selects which examples receive expensive supervision.
2. Its selection score can be entangled with task strata, changing group-level
   acquisition propensities in a fixed pool.
3. This changes the supervision distribution and can affect downstream group
   behavior.
4. DCMS preserves a base utility score while constraining selection over
   observable groups or soft group estimates.

### Working Title

**When Models Choose Their Own Feedback: Model-Induced Acquisition Shift in
Active Preference Learning**

Alternative when multiclass evidence is stronger than DPO evidence:

**Model-Induced Acquisition Shift: Constraining Utility-Driven Supervision
Selection**

The title should follow the strongest completed evidence, not the largest
planned experiment.

## 4. Main-Paper Structure

### 1. Introduction

Use four short moves:

1. Describe the feedback loop: model scores select examples, which become the
   next supervision data.
2. State the overlooked failure mode: score rankings need not be neutral over
   task strata, so the model changes which data it learns from.
3. State MIAS and DCMS in one paragraph, then list two or three contributions.
4. Preview only completed evidence: binary discovery in the appendix, controlled
   multiclass intervention, and fixed-pool preference acquisition when each is
   available.

End the introduction with at most three contributions:

- a falsifiable MIAS measurement and intervention protocol;
- DCMS as a base-selector-agnostic utility-coverage constraint layer;
- evidence across the task families that have passed their experiment gates.

### 2. Problem Setup and MIAS

Define the fixed candidate pool, base acquisition utility, selection budget,
observable group membership, and hidden oracle labels. Separate three objects:
score-stratum dependence, selected-set distribution shift, and downstream
behavior transmission. State that only a declared intervention or randomized
composition comparison supports a causal statement.

### 3. DCMS

Present DCMS as an outer constrained selection problem, not as a new utility
score. Include: rank-normalized base utility, target moments/soft memberships,
utility-retention rule, slack selection, rounding, and recorded propensities.
Keep solver details and robust-interval derivations in the appendix.

### 4. Experimental Protocol

State the common fairness contract once: fixed pools and splits, hidden labels
until selection, shared initialization, equal active-label budget, equal
training-token budget, fixed evaluator, five training seeds for core results,
and paired comparisons. Then give one compact subsection per task:

- binary: IMDb, PAWS `labeled_final`, TweetEval `hate` as native-label evidence;
- multiclass: AG News as the controlled intervention setting and TREC as
  replication;
- preference: HelpSteer2 fixed pairs with length, prompt-cluster, source, and
  A/B-position audits.

### 5. Results

Organize by research question instead of dataset chronology:

1. **Does MIAS occur?** Dependence and Random-calibrated acquisition shift.
2. **Does selector state drive it?** Intervention response curves.
3. **Does composition transmit downstream?** Matched-utility/composition
   comparisons and group metrics.
4. **What does DCMS trade off?** Utility retained, coverage, average quality,
   worst-group metric, and cost.

Each result subsection begins with the claim under test and ends with a scoped
interpretation. A zero or non-replicating result remains a result.

### 6. Related Work

Use four problem-oriented paragraphs: active-learning sampling bias, data
selection for distillation, label/coverage correction, and preference-data
selection plus evaluator bias. Add citations only after programmatic
verification; this blueprint intentionally contains none.

### 7. Limitations and Conclusion

Limitations must cover latent/poorly estimated groups, pilot-label cost,
selector and judge calibration, finite-pool setting, and the fact that an
observed coverage target is not automatically an optimal training distribution.
The conclusion should say that DCMS limits a measured transmission path, not
that it removes bias from a model.

## 5. Figures and Tables

| Item | Claim supported | Minimum completed input |
|---|---|---|
| Fig. 1: MIAS/DCMS overview | The feedback loop and where DCMS acts | Algorithm and protocol only |
| Fig. 2: intervention response | Selector state changes acquisition propensity | Class-intercept and/or length-coefficient sweep |
| Fig. 3: utility-coverage frontier | DCMS changes the trade-off, not just TV | Base selector and paired `+DCMS` runs |
| Table 1: multiclass/binary evidence | Dependence, coverage, and downstream metrics | Completed per-dataset run records |
| Table 2: DPO main result | Preference, worst-group, length-controlled, capability, cost | Completed six-method DPO matrix |
| Table 3: ablation | Robust groups, slack, moment-matched Random, or intervention | Predeclared ablation runs |

Binary historical summaries belong in the appendix until their sample-level
records are regenerated. Do not pool classification and preference metrics into
one average table.

## 6. Abstract Skeleton

Use this only after the evidence contract is satisfied:

> Model-driven acquisition uses a current model to decide which examples receive
> costly supervision. We show that [completed intervention/evidence] can change
> group-specific acquisition propensity in a fixed pool, a phenomenon we call
> model-induced acquisition shift (MIAS). We introduce distribution-constrained
> model-guided selection (DCMS), which [method detail]. Across [only completed
> task families], DCMS [frozen aggregate result with uncertainty] while
> [required trade-off/cost qualification]. These results show [scoped claim].

Never insert a cross-task phrase, a causal verb, or a best-number claim without
its corresponding frozen artifact.

## 7. Writing Order

1. Freeze the evidence contract and produce Fig. 1 from the actual algorithm.
2. Complete task-specific run records and build result figures/tables.
3. Write Methods and Experimental Protocol from frozen configs.
4. Write Results around the four research questions.
5. Draft the Introduction and Abstract last, choosing the title from the
   strongest evidence.
6. Add verified related-work citations, limitations, appendix details, and the
   venue checklist.

This order prevents the manuscript from hardening around results that the
experiments have not yet established.
