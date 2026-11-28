# Biomedical-pool feasibility pilot

The project owner approved piloting Qwen3-8B, BioDecision-v2-4B and
MedGemma-27B-text-it with Qwen3 aggregation before any full profiling or
evaluation. All new pilot/profiling allocations count toward eight GPU-hours.

- `biomedical_pilot.py`: isolated per-model workers; fixed development and
  seeded training inputs; explicit revisions; output and timing records.
- `biomedical-pilot.sbatch`: 30-minute, one-A100 Slurm allocation; configurable
  repository, container and cache roots through `SKILL_MOE_ROOT`,
  `SKILL_MOE_IMAGE`, `SKILL_MOE_HUB`.
- `gpu-budget.json`: allocation accounting; overlapping Slurm steps are not summed.
- `summarize_pilot.py`: measured-throughput scenarios and remaining budget.
- `runs/pilot-41597961/RESULT.md`: first pilot results and proposed replacement.

First pilot status: all models loaded, but strict answer-format coverage was
incomplete, GPU-memory sampling failed, and current throughput estimates exceed
the budget. The memory UUID fix is in the working script; original executed code
is preserved in the run directory. **No full profiling/evaluation is authorized
by a successful process exit alone.** Re-pilot and resolve pool/throughput issues.

The owner subsequently approved a replacement pilot: Qwen3-8B, BioDecision-v2-4B,
and QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ, with Qwen3 aggregation.
Job `41599259` tests compiled execution, 32 concurrent sequences, a 32-row seeded
training timing batch, five exposed development questions, concise reasoning and
explicit plain-text final-answer instructions. The current batch script runs
this replacement pilot. Its results must be checked before fixing the pool.

Current inference workers contain no gold labels in their inputs. Profiles will
use training gold answers for scoring only after a viable pool is fixed. Routing
must use query text/keywords and actual normalized profile-derived probabilities;
evaluation gold answers remain scoring-only. Do not promote this pilot to a
primary result or use existing same-model historical configs as launch approval.

Regenerate the first pilot summary from repository root:

```bash
python3 experiments/summarize_pilot.py experiments/runs/pilot-41597961 --allocated-seconds 1068
```

## Routed collection on Cheaha

The replacement pilot passed the feasibility check. The pinned Qwen3-8B,
BioDecision-v2-4B, and Qwen3-Coder-A3B-AWQ pool uses Qwen3-8B as aggregator.
The deployed source, data, split manifest, and MiniLM encoder were SHA-256
verified in `~/skillmoe-routed-v1`. Run artifacts live under
`experiments/runs/routed-primary` in that directory.

- Profiling job `41603728`: completed, 752 GPU-seconds charged; all 504
  training rows have keywords, and all three checkpoint-matched profiles
  were independently recomputed from their training outputs. Strict parse
  counts were 504/504 Qwen, 496/504 BioDecision, and 504/504 Coder.
- Recruited smoke job `41603767`: completed, 570 GPU-seconds charged. All
  15 expert and five aggregator responses parsed. Draw counts were 3 Qwen,
  9 BioDecision, and 3 Coder; actual router confidence ranged from 0.5290
  to 0.7813. Repeat execution preserved all response caches, routes, and
  final smoke records byte-for-byte. Both scoring paths got 5/5 on the
  five previously exposed examples; this is a development check.
- Primary job `41603829`: submitted for all 4,178 frozen non-smoke questions.
  The allocation has a two-hour limit; `~/skillmoe-routed-v1/primary-41603829.log`
  and `experiments/runs/routed-primary/stages/evaluate-*.log` track progress.

The collector is a Skill-MoE adaptation with deterministic per-request seeds,
concise prompts, and a pinned three-model pool. Collection produces response and
routing records; primary labels are joined later for the planned paired analysis.
