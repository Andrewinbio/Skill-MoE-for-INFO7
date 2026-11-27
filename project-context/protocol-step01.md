# Step 01 Protocol: Scope and Evaluation

Status: **Active Step 01 protocol, reconciled against the professor's feedback;
methodological decisions listed below remain open.** Scope of this revision is
protocol review only. It does not implement routing, submit jobs, or proceed to
Step 02.

## Source precedence and proposal reconciliation

The current proposal is `proposal.md`, including its same-model amendment.
Where that proposal conflicts with `professor-feedback.md`, **the professor's
feedback controls**. `rubric.md` controls course deliverables and execution,
documentation, contribution, and presentation requirements.

The proposal's amendment removes router-confidence comparisons, while its
research-question and methods sections retain them. The professor explicitly
requests comparing disagreement with router confidence. Therefore:

- **Restore router confidence as a required comparator**, with router-only,
  disagreement-only, and combined predictive models.
- Use genuine probabilities from the expert recruitment process. A fixed
  single-model fixture with confidence identically one cannot answer this
  comparison. Prior Qwen3-only runs remain development/smoke evidence.
- Plan a verified multi-model candidate pool with three sampled slots, allowing
  repeats. The professor does not specify particular checkpoint identities or
  require three distinct experts on every question; those details need input.
- Retain paired expert-vote/aggregator outcomes, two disagreement measures,
  probability calibration, accuracy and skip-rate reporting, and the required
  always-aggregate, always-vote, and random-bypass baselines.

This Step 01 protocol supersedes the conflicting same-model-only scope in
`primary-protocol-final.md` and `../Skill-MoE-for-INFO7/analysis/primary-study.json`.
Those earlier artifacts are not a valid launch specification for the restored
study and have not been rewritten or executed in this Step 01 task.

| Professor feedback | Protocol response |
|---|---|
| Assess probability calibration using the shared literature | Separate discrimination from calibration; review the cited references before selecting the calibration procedure |
| Study when aggregation adds value on paired questions | Helpful/harmful/failure labels and paired vote-versus-aggregation scoring |
| Account for repeated selections and coarse disagreement | Preserve draw order, count distinct experts, add entropy, and report distinct-expert sensitivity |
| Compare disagreement with router confidence | Required router-only, disagreement-only, and combined feature sets |
| Compare adaptive inference with fixed and random baselines | Always aggregate, always vote, matched-call-rate random bypass, plus raw-disagreement threshold |
| Consider MedMCQA and MMLU-Pro and create question-level data | MedMCQA first; record why MMLU-Pro is deferred, with an extension contingent on budget |

## Research question

Can expert disagreement predict aggregation failures and enable calibrated,
cost-efficient inference in Skill-MoE?

The experiment tests whether information available after expert generation but
before aggregation predicts when aggregation is helpful, and whether that
signal can reduce aggregator calls without exceeding a pre-specified accuracy
loss relative to always aggregating.

## Initial scope

The proposed first benchmark is **MedMCQA**, because it is a biomedical,
multiple-choice benchmark and the local upstream copy is available. The
initial unit of analysis is one question. The experiment will use the existing
Skill-MoE workflow rather than redesigning the router:

- Three expert slots are selected by the upstream recruitment procedure.
- The selected models and their outputs are retained in draw order; repeated
  selections are not silently deduplicated.
- Expert voting is the modal normalized answer across the selected slots.
- The aggregator identity and exact checkpoint revision must be confirmed for
  this restored study; the symbolic name `Qwen` alone is insufficient. The
  previously tested Qwen3-8B is a candidate, not automatically a reproduction of
  the upstream Qwen2.5 configuration.
- Retain the previously approved seed `0` unless explicitly revised.
- No profiles or configurations are generated in Step 01. Later execution
  needs profiles matched to the exact expert checkpoints, plus recruitment
  inputs whose availability has been verified.

The existing local audit and smoke artifacts are prior work, not evidence that
router-based experiments have run. External dataset provenance, near-duplicates,
and model training overlap remain limited. MedMCQA is the retained first
benchmark; the professor asks that MedMCQA and MMLU-Pro be considered, not that
both necessarily be run. Adding MMLU-Pro needs a budget/scope decision.

## Outcomes and labels

Let `V` be the expert-vote answer, `A` the aggregator answer, and `Y` the
reference answer. Both answers are parsed with the same benchmark-appropriate
answer parser and scored against `Y`.

- **Aggregation failure:** `A` is incorrect. This is the primary failure
  outcome for the aggregation model.
- **Helpful aggregation:** `V` is incorrect and `A` is correct. This is the
  decision target for invoking the aggregator.
- **Harmful aggregation:** `V` is correct and `A` is incorrect. This is the
  principal risk of invoking the aggregator.
- **Neutral agreement:** `V` and `A` have the same correctness status. This
  includes both correct and incorrect cases where aggregation does not change
  the outcome.
- **Net gain:** aggregation is correct while voting is incorrect, coded as a
  positive change; harmful aggregation is coded as a negative change.

Invalid or unparsable answers count as incorrect for scoring and do not vote.
The raw output, parsed answer, and parse-status flag must be retained. A missing expert or
aggregator output is not converted into a correct answer; incomplete question
records are reported separately and excluded from paired analyses that require
both predictions. Any exclusion rule used in the final analysis must be
reported with coverage.

For a vote tie, choose the answer from the earliest selected expert slot among
the tied answers. This deterministic rule must be applied before looking at the
reference answer. A sensitivity analysis will give each distinct selected
expert one vote and will use the earliest draw for that expert if repeated
draws disagree.

## Predictors available before aggregation

The question-level record will contain router and expert information, but the
adaptive policy may use only fields available before the aggregator call:

- Router confidence: maximum probability over the complete candidate-expert
  selection distribution used to draw the slots for that question. Capture the
  model-to-probability mapping **before sampling**, its normalization and any
  fallback behavior. Preserve zero-probability candidates where applicable.
  These probabilities describe selection, not probability of answer correctness.
  Do not substitute ranks, raw unnormalized scores, empirical draw frequencies,
  aggregator confidence, or a fabricated constant. If upstream uses a different
  operational definition, document and settle that definition before collection.
- Modal disagreement: `1 - (largest valid answer count / number of valid
  expert draws)`.
- Normalized answer entropy: Shannon entropy of valid answer proportions,
  divided by `log(number of available answer choices)`. It is zero when there
  is one valid answer category. If there are no valid expert answers, entropy
  and modal disagreement are missing and the record is flagged.
- Number of selected slots and number of unique selected experts.
- Optional timing and token counts, used for cost reporting rather than
  policy decisions unless explicitly approved later.

Aggregator output, reference answer, and correctness labels are never policy
inputs. Router confidence alone, disagreement features alone, and a combined
feature set will be compared in later calibration work.

Profiles, router weights, keyword mappings and preprocessing must not be fitted
or selected using validation/held-out correctness labels. Record their source
data and construction procedure. Query-text-based routing is allowed, but using
the query's gold answer to create a router feature is not.

Required question-level fields are stable ID, source/split, question, reference
answer, candidate expert IDs and full routing distribution, confidence definition,
selected slots in draw order, raw expert and aggregator responses, parsed answers,
parse/finish status, vote/tie status, pre-aggregation features, and outcome labels.
Record checkpoint revisions, profile provenance, run settings and stage costs
when available. Missing router information must remain explicitly missing;
record coverage and do not silently turn the required comparator into a
disagreement-only experiment.

## Data partition and evaluation

Retain the previously approved seed-0, duplicate-grouped MedMCQA manifest:
2,507 fit / 831 validation / 840 held-out questions, with five smoke-exposed
questions reserved for development. The intended primary sample remains all
4,178 non-smoke questions. This is a sampling target, not a claim that the
restored multi-model run will fit the compute budget. Any sample-size change
must be approved and frozen before observing primary outcomes.

The fit split is used for logistic-model coefficients and feature preprocessing,
the validation split for policy threshold selection, and held-out questions once
for final evaluation. No held-out outcome is used for model or threshold selection.
If additional probability recalibration is used, reserve data or use out-of-fold
fit predictions rather than fitting the calibrator on held-out data.

Fit separate models for aggregation failure and helpful aggregation with each
of three feature sets: router confidence only, disagreement/entropy only, and
router confidence plus disagreement/entropy. Slot counts, invalid-answer flags
and distinct-expert counts are retained for diagnostics and sensitivity analyses;
whether to add them as covariates must be specified consistently before fitting.

The final analysis will compare:

1. Always aggregate.
2. Always use expert voting.
3. Random bypass with the same aggregator call rate as the adaptive policy.
4. A raw-disagreement threshold.
5. A calibrated policy that calls the aggregator when predicted helpful-
    aggregation probability exceeds the validation-selected threshold.

Retain the previously approved **2-percentage-point absolute accuracy-loss
tolerance**, relative to always aggregating. Proposed threshold selection is
maximum skip rate subject to that validation constraint; freeze the exact
threshold candidates, tie-break, and all-invalid handling before tuning. Report
held-out violations rather than retuning to hide them. An empirical tolerance
is not a statistical noninferiority guarantee. Define the matched-call-rate
randomization procedure and seeds in advance, independently of outcomes.

The adaptive policy is evaluated as replay unless a later implementation step
verifies that the live path actually skips the aggregator. Expert generation
remains a cost of every policy. When direct runtime or token measurements are
unavailable, skipped aggregator calls are reported as a call-count cost proxy,
not as total inference cost.

## Metrics

Primary policy metrics on held-out questions:

- Answer accuracy for every policy.
- Absolute accuracy difference from always aggregate.
- Aggregator call rate and skip rate.
- Counts and rates of helpful and harmful aggregation.
- Paired question-level differences between voting and aggregation.

Primary prediction and calibration metrics:

- AUROC and AUPRC for aggregation failure and helpful aggregation, reported
  separately.
- Brier score, event prevalence, and reliability plots.
- Coverage and counts of invalid, missing, and tied expert votes.

Timing and token counts will be reported by stage when the implementation makes
them available. Results must identify the benchmark, seed, upstream revision,
model/checkpoint identifiers, split counts, and run settings.

Because three draws produce only a few disagreement levels, report their
distribution and the number of unique experts. Do not interpret adding entropy
as automatically providing independent predictive information. For rare or
single-class outcomes, report prevalence and undefined discrimination metrics
rather than fitting or interpreting an unsupported model.

Required calibration reading before methodological lock:

- [JAMIA calibration reference supplied by the professor](https://academic.oup.com/jamia/article/27/4/621/5762806)
- [Scikit-learn probability calibration guide supplied by the professor](https://scikit-learn.org/stable/modules/calibration.html)

These references motivate assessing probabilities, not just classification
accuracy. Step 01 records the reading requirement; it does not claim a completed
literature review or select an additional recalibration method by default.

## Compute budget

Retain the previously approved **8 allocated GPU-hour cap**, with one A100 at
a time on Cheaha using batch jobs. Primary startup, failures and retries count;
previous completed smoke jobs do not. Additional profile-generation or routing
GPU allocations must also be accounted for within this restored-study cap unless
the owner explicitly approves a separate budget. CPU-only analysis is separate.

Whether the verified multi-model pool and all 4,178 questions fit the cap is
unresolved. If collection cannot finish, report incomplete coverage rather than
silently dropping router confidence, changing the sample based on results, or
extending the budget. Confirm whether new profiles may be generated if existing
profiles cannot be obtained. No new experiment result is claimed by Step 01.

## Code ownership

The supplied project materials identify one actual team member: **Andrew
DePass**. His planned ownership is the protocol and experimental design,
question-level paired-output schema, disagreement and calibration analysis,
reproducibility documentation, and poster methods/results documentation.

No other team members are named in the supplied files. If there are teammates,
record their actual names and meaningful code responsibilities; do not invent
members or imply a solo project requires additional members. Each actual member
needs code commits and a documented contribution statement, not only presentation
work.

## Rubric alignment and deliverables

This is an aligned research plan, not a claim that the submission is complete.

- **Beyond a tutorial:** the contribution is the paired question-level dataset,
  router/disagreement comparison, calibration analysis and selective aggregation
  evaluation, with original code and appropriately limited conclusions.
- **Runnable GitHub project:** provide source, environment/data/checkpoint setup,
  command-line or notebook instructions, and evidence that the complete workflow
  runs on Cheaha. Successful smoke execution does not prove primary readiness.
- **Organization and portability:** configurable paths, clear project structure,
  and descriptive notebook cell titles if notebooks are used. Account-specific
  development paths must not be the only way to reproduce results.
- **README:** user-facing setup, data/model acquisition, execution and analysis
  instructions for this project, beyond the upstream README.
- **Contributions:** meaningful code-commit history and contribution descriptions
  for every actual team member.
- **Poster:** electronic PowerPoint or PDF plus printed copy; problem, methods
  tried, results so far, GitHub URL, and actual contributions. Present in person
  unless officially exempted; all members should be able to discuss results.
- **Course overlap:** disclose and obtain the required instructor approvals if
  there is overlap with another course; applicability remains a user question.
- **Dates:** rubric lists December 2 for the poster session and December 4 for
  final code commits. Verify any later course announcements separately.

The rubric gives methods/results 30% for master's and 40% for PhD students,
documentation/organization 10%, README 10%, and poster delivery 40%; Cheaha
execution is 10% for master's and required for PhD. It also notes inconsistent
50% presentation wording in the handout. Do not resolve that discrepancy or
assume the student's degree category without confirmation.

## Decisions required before experiments

The professor requires the comparisons but does not settle every operational
choice. The following need user input before primary collection or fitting:

1. **Expert pool and matched profiles:** which model checkpoints/revisions should
   be eligible? May unavailable profiles be regenerated, and from which data?
   The single-model smoke fixture cannot be promoted to the primary router study.
2. **Aggregator:** retain the tested Qwen3-8B or stage the upstream Qwen2.5
   checkpoint? Freeze the revision, thinking mode, generation cap and temperature
   for the restored configuration. Profiles must not be relabeled across models.
3. **Router confidence semantics:** approve maximum normalized per-question
   recruitment probability as the operational definition, or specify another
   genuine upstream confidence quantity. If the actual router offers only
   constant/global weights, decide how to report or address that limitation;
   do not invent query-specific confidence.
4. **Feasibility trade-off:** retain the approved all-question target within
   8 GPU-hours if feasible; otherwise explicitly choose staging more resources,
   a smaller predeclared sample, or a revised cap. Do not drop the professor's
   router comparison as an implicit compute shortcut.
5. **Calibration specification:** confirm logistic regression alone with
   calibration assessment versus an additional calibrator trained on a dedicated
   calibration subset or out-of-fold fit predictions, plus regularization and
   optional covariates. If added, approve how calibration data are separated from
   threshold selection and final evaluation. The shared reading informs this.
6. **Decision rules:** confirm earliest-draw ties, strict explicit-answer parsing,
   first-draw distinct-expert sensitivity, all-invalid forced aggregation, and
   validation threshold/random-baseline details before tuning. These are proposed
   operational rules, not requirements explicitly dictated by the professor.
7. **Second benchmark:** is MMLU-Pro an optional extension or a required additional
   run? MedMCQA remains the first benchmark; the feedback does not require both.

Previously approved seed 0, the existing MedMCQA split and smoke exclusions,
8 GPU-hour cap, and 2-percentage-point tolerance are carried forward rather
than falsely described as never approved. Any change to those choices must be
explicit. Additional administrative input: actual team composition, degree
category for the rubric, and whether cross-course overlap applies.

## Step 01 completion boundary

This document saves the conflict resolution, planned methods, rubric mapping and
open decisions. Stop here. Code changes, router instrumentation, checkpoint/profile
staging, configuration migration, dataset changes and job submissions require
a subsequent authorized step. Earlier same-model configuration artifacts remain
historical and must not be used as a router-study launch approval.
