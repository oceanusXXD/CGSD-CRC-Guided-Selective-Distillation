# Experiment Report Status

The reports in this directory are retained as binary-task evidence and provenance. They are not the canonical AAAI paper narrative.

- CRC-error-mass, NS-error-mass, and Defer-kcenter should be interpreted as legacy probes that expose selector-induced supervision shift.
- PCSS results are preliminary evidence for the binary two-stratum special case of DGA; they do not establish the final method.
- Legacy binary-task result tables, figures, and the old result summary live in `binary_legacy/`.
- Current DPO planning/status artifacts live in `dpo_main/current/`; these are execution readiness and partial stage-execution files, not completed DPO results. The Random selection/reveal branch has produced selected/revealed/train-row artifacts for seeds 1-5. The manifest now contains training, evaluation, and run-record commands, while actual training remains blocked by missing usable model checkpoint paths and actual evaluation remains blocked by missing held-out prediction, judge, and capability inputs. All logprob-dependent baseline/DCMS branches remain blocked by missing true logprobs and model checkpoints.
- Current multiclass and preference protocols are documented in `../benchmark_shift_protocol.md`.
- The canonical paper direction is `../../docs/aaai_long_paper_direction.md`.
