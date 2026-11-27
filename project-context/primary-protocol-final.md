# Final primary protocol — Qwen3 sampled-response aggregation

Status: **Finalized with explicit project-owner approval in this session.**
This supersedes the provisional experiment scope in `protocol-step01.md`.
Machine-readable source: `Skill-MoE-for-INFO7/analysis/primary-study.json`.
No primary experiment has been submitted by this finalization step.

## Approved scope and research question

Can disagreement among three sampled responses from **one Qwen3-8B model**
predict aggregation failure or benefit, and reduce aggregator calls within
an absolute accuracy loss of **2 percentage points**?

This is a same-model sampled-response study using Skill-MoE-derived inference
and aggregation prompts. It is not a reproduction of heterogeneous recruitment
or the original router-confidence hypothesis. No router exists in this fixed
three-draw design; router-confidence-only and combined-router comparisons are
removed. Unique-expert count is always one and is reported, not fitted as an
informative predictor. Distinct-expert sensitivity reduces to the first draw.

## Frozen data and model

- Benchmark: the local upstream MedMCQA files, pinned by SHA-256 in the config.
- Primary sample: **all 4,178 non-smoke questions**.
- Fixed manifest: **2,507 fit / 831 validation / 840 held-out**.
- Five already inspected smoke questions remain development-only.
- Seed: **0**. Normalized question IDs and split-group assignments are frozen.
- Checkpoint: **Qwen/Qwen3-8B**, revision
  `b968826d9c46dd6066d109eabc6255188de91218`, for both sampled responses and aggregation.
- Three separately sampled responses per question in retained draw order.
- Non-thinking mode, **4,096-token cap**, temperature **0.7** for both stages.
- Explicit sampling defaults for primary collection: top-p 1, top-k -1,
  min-p 0, repetition penalty 1. Do not silently inherit model defaults.
- Prompt source: the successful quality-run source copies in
  `smoke-runs/medmcqa-step03/substitute-41597473/`, hashed in the config.
- Backend: vLLM on the verified Cheaha container. Record actual package versions,
  model revision, prompt hashes, sampling parameters and execution/chunk seeds
  in the primary run manifest before collecting primary data.

External benchmark lineage, near-duplicates, and checkpoint training overlap
remain limitations. Local hash/duplicate checks do not establish their absence.

## Compute contract

Use Slurm batch jobs on one A100 80 GB at a time, with 8 CPUs and 48 GB host RAM.
**The total primary budget is 8 allocated GPU-hours**, including startup,
failed primary attempts and retries. Earlier development smoke jobs are excluded.
Each allocation is limited to at most 2 hours; jobs are sequential, not parallel.
Queue waiting time does not count. CPU-only analysis does not consume GPU budget.

The primary collector must checkpoint complete question records, avoid duplicate
generation on resume, and maintain a cumulative Slurm allocation ledger. Each
new job's time limit must not exceed the remaining budget. Do not assume four
jobs automatically enforce the cap if failures or retries also occur.

If the cap is reached, stop and report incomplete collection with IDs and
coverage. Do not silently reduce the sample based on results or increase the
budget. The cap is not a promise that all questions will fit within it.

## Parsing, features and labels

Use the strict parser and draw-order rules documented in `analysis/README.md`.
Length-limited responses and unfinished thinking are invalid; invalid answers
count as incorrect and do not vote. Missing outputs are reported and excluded
from paired metrics; full primary conclusions require complete collection.

Fit predictors: modal disagreement, normalized answer entropy, valid expert
answer count, and a no-valid-answers indicator. Replace undefined disagreement
and entropy with zero only alongside that indicator. Do not use reference
answers, aggregator outputs, correctness, or post-aggregation timing as predictors.

Separate prediction targets are aggregation failure and helpful aggregation.
Also report harmful aggregation, neutral outcomes and net correctness gain.
All-invalid expert answers force an aggregator call for adaptive policies.

## Fixed modeling and threshold selection

Fit a StandardScaler and L2 logistic regression separately for each target,
using only the fit split. Parameters: C=1, lbfgs solver, no class weighting,
max_iter=1,000, seed 0. No hyperparameter search. If a target has only one fit
class, use its constant prevalence, mark discrimination metrics undefined,
and do not claim informative calibration. Unusable fits fall back to always
aggregating; diagnose numerical failures without consulting held-out outcomes.

The helpful-aggregation probability drives the adaptive policy. Select its
threshold on validation data to **maximize skip rate**, subject to validation
accuracy at least `always_aggregate_accuracy - 0.02`. Candidate thresholds are
unique validation scores and always-call/always-bypass endpoints. The rule is
`call if score > threshold`, with the all-invalid override. Break ties by higher
validation accuracy, then larger threshold. If no threshold is feasible, always
aggregate. Select the raw-disagreement threshold by the same procedure.

Freeze fitted models, thresholds, preprocessing and evaluation code before
inspecting held-out outcomes. Evaluate held-out once. A held-out tolerance
violation is a result to report, not permission to retune. The 2-point tolerance
is an empirical constraint, not a formal statistical noninferiority guarantee.

## Comparators, metrics and uncertainty

Compare always aggregate, always vote, raw-disagreement threshold, calibrated
helpfulness policy, and random bypass matched to the calibrated policy's call
count. Random bypass uses the same forced calls plus uniformly sampled remaining
calls, with 100 predetermined seeds 0–99; report the distribution, not a selected
best random seed. Match call counts without consulting correctness labels.

Report prediction AUROC, AUPRC, Brier score, prevalence and 10 equal-width-bin
reliability plots. Report policy accuracy, accuracy difference from aggregation,
call/skip rates, helpful/harmful counts, ties, invalid/missing counts and coverage.
For held-out paired accuracy differences, use 95% question-level paired bootstrap
intervals with 2,000 resamples and seed 0. Note undefined metrics when an evaluation
target contains only one class.

Policies are evaluated by **replay**. Call savings are a proxy, not demonstrated
wall-time savings. Expert generation is charged to every policy. Measured
generation tokens/times are reported separately from full allocation runtime.

## Ownership and launch readiness

Andrew DePass remains the only identified contributor. No teammate, access
approval, or coursework-overlap determination is invented by this document.

Run `python3 analysis/validate_primary.py` from the repository to verify the
frozen data, split, source hashes and quality-run evidence.

**Protocol finalized; production execution remains to be implemented.** The
existing smoke runner deliberately keeps only five questions. Next implement
the resumable all-question collector, explicit sampling/runtime manifest and
budget ledger, followed by the frozen fit/validation/replay evaluator. These
must honor this config rather than treating the smoke script as the primary job.
