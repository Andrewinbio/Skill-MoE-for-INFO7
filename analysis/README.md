# MedMCQA preparation and paired-output analysis

Standard-library tooling; run commands from the repository root.

## Active router study and historical configuration

`project-context/protocol-step01.md` is the active protocol: router-only,
disagreement-only and combined models for both aggregation failure and helpful
aggregation. `primary-study.json` is the superseded same-model configuration,
not launch approval. See `experiments/README.md` for the ongoing multi-model
feasibility pilots and `experiments/gpu-budget.json` for the eight-hour ledger.
The existing 2,507/831/840 split, seed 0, development exclusions and 2-percentage-
point accuracy-loss tolerance remain applicable. Pool feasibility is still being
tested before full profiling and primary collection.

`recruit_agents.py --log-router` now records the actual normalized sampling
distribution, maximum probability and fallback flag in its solver CSV. The flag
is opt-in and preserves the draw sequence and RNG state. Annotation and matched
profile inputs must exist before this command can run. The analysis consumes
these CSV fields, validates the probability vector, and stores it in each record.
Manual smoke fixtures continue to have null router information.

```bash
python3 -m unittest discover -s analysis -p 'test_*.py'
python3 analysis/prepare_medmcqa.py dataset \
  --test test_data/MedMCQA_test.json \
  --train test_data/MedMCQA_train.json \
  --exposed-csv smoke-runs/medmcqa-step03/run/fixture_medmcqa_smoke.csv \
  --output analysis/medmcqa-data
python3 analysis/prepare_medmcqa.py records \
  --run-dir smoke-runs/medmcqa-step03/substitute-JOBID \
  --splits analysis/medmcqa-data/split_manifest.json
```

## Dataset audit and provisional splits

Question IDs are SHA-256 hashes of NFKC-normalized, case-folded question text
with whitespace collapsed. Choices are already embedded in upstream question
text. Duplicate rows share an ID and split. Label conflicts and exact-normalized
train overlap are excluded from experimental splits. Smoke-exposed questions
are reserved as `development_smoke`, never held out. Add all previously exposed
questions to the exposure CSV before freezing a primary-run split.

Split assignment hashes `seed:question_id` into provisional 60/20/20 ranges;
counts are approximate proportions rather than exactly sized samples. This
is stable under row reordering and adding unrelated questions. Seed defaults
to 0. The manifest stores source-row indices and exclusion/exposure flags.
Dataset files and exposure CSV are hashed in the audit. External provenance,
near-duplicates, and checkpoint training contamination remain unverified.

## Question-record schema (version 1.1)

`question-records.jsonl` contains one record per CSV row:

- `question_id`, `source_index` (row within this run), `run_id`, `split`,
  `question`, `gold_answer`.
- `experts`: ordered draw list with `slot` (zero-based), solver, source column,
  raw output, normalized answer, parse status, model, prompt/output token counts,
  finish/stop reason, batch generation duration, token cap, and thinking mode.
- `aggregator`: the same response fields.
- `router`: full candidate probabilities, maximum probability, and fallback
  provenance when logged. Probabilities must sum to one and each selected
  expert must have positive probability. Confidence is computed over the full
  distribution, including experts not selected in the three draws.
- `features`: only pre-aggregation information: modal disagreement, normalized
  entropy, selected/unique expert counts, valid/invalid answer counts, and
  nullable router confidence. Manual fixtures have no router probabilities;
  confidence is never invented.
- `vote`: modal valid answer, tie flag, and distinct-expert sensitivity vote.
- `labels`: vote/aggregation correctness, aggregation failure, helpful,
  harmful, neutral, and net gain. These and the gold answer must never be
  policy inputs.
- `paired_eligible`: false when any selected expert or aggregator output is
  missing. Missing records are excluded from paired metrics with coverage
  reported; present but invalid/truncated outputs count as incorrect.

The upstream writer labels repeated draws in reverse (`answer_N` first), so
column order is not draw order. We reconstruct the draw order before voting.
Ties select the earliest valid draw among tied answers. Distinct-expert voting
retains the first draw for each expert, even when that first draw is invalid.

## Strict parsing and quality checks

Require `answer is (A)` through `(D)`, case-insensitive, outside a completed
thinking block. The last explicit answer wins. A `length` finish reason is
invalid, even if an earlier explicit answer appears. Unclosed thinking blocks,
missing text and absent explicit answers receive separate statuses. No letter
is guessed from reasoning. Older runs without generation telemetry have null
token/finish fields; truncation cannot be established solely from token count.

Modal disagreement is `1 - max(valid counts)/valid draws`; normalized entropy
uses `log(4)`. Both are null when no valid expert answers exist. The summary
reports that coverage, ties and parse failures alongside accuracy and outcomes.
Telemetry is matched against raw responses and rejected on missing, duplicate,
unmatched, or mismatched events. Aggregation/reference/CSV row counts must match.

## Timing and cost

`generation-events.jsonl` is generated by the isolated vLLM smoke copies.
The timer surrounds `llm.generate`; it excludes model loading and most startup.
Batch duration is repeated per response for association, **not per-question
latency**. Summary durations count each solver batch once. Token totals are
measured generated/request tokens, not dollar cost; expert generation remains
necessary even when a future policy bypasses aggregation. Original batch logs
contain full job runtime. `analysis-summary.json` hashes input artifacts.

## Scope

This implements preparation and smoke diagnostics, not calibrated policy fitting.
Use only `features` as predictors and honor fit/validation/held-out boundaries
after split approval. Five development questions and repeated draws from one
model cannot establish calibration or heterogeneous-expert performance.
The historical same-model settings in `primary-study.json` are superseded. A
resumable primary collector and fit/validation/replay evaluator still need
implementation; the multi-model feasibility budget is tracked separately.
The dataset audit's
"provisional" wording records its creation state; the exact manifest hash is
now approved without rewriting that historical artifact.

## Routed primary collection scoring (Steps 06–07)

Use the structurally validated `evaluate-records.json` from Cheaha job
`41603829`. The original run artifacts stay in
`~/skillmoe-routed-v1/experiments/runs/routed-primary/` on Cheaha.
From the repository root, with the Mac connected to Cheaha:

```sh
mkdir -p experiments/runs/routed-primary
scp cheaha:skillmoe-routed-v1/experiments/runs/routed-primary/evaluate-records.json experiments/runs/routed-primary/
python3 analysis/score_routed.py \
  experiments/runs/routed-primary/evaluate-records.json \
  experiments/runs/routed-primary/fit-validation
```

The output directory must be new. `features.csv` contains identifiers,
pre-aggregation features, and vote diagnostics. `outcomes.csv` contains
paired eligibility and outcome labels, keyed by question ID, source index and
split. `summary.json` records source hashes, coverage, repeats, ties, and
the paired correctness table. A generated but invalid response scores
incorrect and contributes no expert vote. An absent response, recorded as
`missing`, excludes the row from paired outcome analysis. Blank feature
values indicate that no valid expert answer was available. The distinct-expert
analysis uses each solver's first draw, even when that draw is invalid.

The scorer processes only fit and validation records. Freeze the policy
using validation before enabling held-out scoring in a separate step.
