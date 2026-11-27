# Step 03 substitute-model smoke result

Job `41597435`: **COMPLETED**, exit `0:0`, elapsed **3m 52s** (2026-11-27 UTC).

Qwen/Qwen3-8B, cached snapshot `b968826d9c46dd6066d109eabc6255188de91218`,
upstream vLLM backend, one A100 80 GB, 8 CPUs, 48 GB host RAM, seed 0,
1,024-token cap. This is the user-approved substitute smoke run, not the exact
Qwen2.5 protocol or a heterogeneous-expert experiment.

Validation: five aligned reference/CSV rows, 15 nonempty expert responses,
five nonempty aggregator responses; log ends with `smoke test completed`.

| Row (zero-based) | Gold | Expert predictions | Majority | Aggregator |
|---|---|---|---|---|
| 0 | A | A, B, B | B | A |
| 1 | A | C, A, D | C | D |
| 2 | C | C, C, C | C | C |
| 3 | C | C, C, C | C | C |
| 4 | A | A, A, A | A | A |

Upstream-parser accuracy: majority **60%**, aggregation **80%**.
Nine expert responses and one aggregator response lack an explicit
`answer is (X)` pattern; inspected examples end mid-reasoning. The short cap
and permissive fallback parser make these scores unsuitable as benchmark
accuracy. The end-to-end execution smoke check passed; answer-completeness
quality remains a follow-up.

`slurm.log` is the original batch log. The Python files here are the executed
isolated copies. `test_data/MedMCQA_test.json` contains only the five evaluated
questions, keeping the scoring denominator aligned. Output files are under
`skills/MedMCQA/`. Batch configuration: `../cheaha-substitute.sbatch`; runner:
`../run_smoke.sh` (staged remotely as `/home/avdepas2/skillmoe-run_smoke.sh`).
