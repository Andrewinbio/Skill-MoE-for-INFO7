# Answer-quality smoke validation

Slurm job **41597473**, **COMPLETED**, exit **0:0**, elapsed **4m 18s**.
Run date: 2026-11-27 UTC. One A100 80 GB, 8 CPUs, 48 GB RAM.

User-approved substitute `Qwen/Qwen3-8B`, snapshot
`b968826d9c46dd6066d109eabc6255188de91218`, seed 0, five MedMCQA questions
with three Qwen draws each. **Non-thinking mode**, maximum **4,096 tokens**.
This changes both thinking mode and generation cap from the prior smoke run;
it is a quality configuration check, not an ablation of either setting alone.

## Validation

- 15/15 expert responses and 5/5 aggregation responses have explicit answers.
- All 20 finish reasons are `stop`; none is truncated.
- Raw generation telemetry matches every saved response exactly.
- Five question/reference/CSV records align; paired coverage is 100%.
- Strict expert-vote accuracy: 4/5 (80%).
- Strict aggregator accuracy: 4/5 (80%).
- Helpful: 0; harmful: 0; neutral: 5. No vote ties or all-invalid expert sets.
- Expert generation: 2,517 prompt tokens, 7,432 output tokens, 10.50 seconds.
- Aggregator generation: 8,716 prompt tokens, 2,679 output tokens, 8.66 seconds.

Generation timings exclude model initialization and are batch timings, not
per-question latency. These five development examples are not benchmark or
calibration evidence. The original Qwen2.5 protocol remains unverified.

## Artifacts

- `generation-events.jsonl`: measured tokens, finish reasons, durations/config.
- `question-records.jsonl`: strict parsed records, original draw-order voting,
  disagreement/entropy features and outcome labels.
- `analysis-summary.json`: validation metrics and artifact hashes.
- `skills/MedMCQA/round0_seed0.csv`: raw expert responses.
- `skills/MedMCQA/fixed_Qwen_round1_seed0_80.0.json`: raw aggregation responses.
- `slurm.log`, `smoke-test.log`: full logs.
- Python sources and `run_smoke.sh`: isolated execution sources and staged runner.

Analysis implementation and schema: `../../../analysis/README.md`.
All five questions are reserved as `development_smoke` in the provisional
dataset manifest and excluded from fit/validation/held-out sets.
