# MedMCQA Smoke Fixture

This fixture contains the first five real questions from
`test_data/MedMCQA_test.json`, with their upstream `gold_answer` values and
three manually assigned solver slots per question.

The only assigned expert is `Qwen`, repeated in all three slots. This is an
intentional minimal configuration: it exercises the upstream three-slot,
with-replacement CSV path while requiring only one Qwen-family checkpoint. The
local smoke script defaults that label to `Qwen/Qwen2.5-1.5B-Instruct` for
Apple Silicon memory limits; set `SKILL_MOE_MODEL_Qwen` to
`Qwen/Qwen2.5-7B-Instruct` to use the upstream-sized model. It is not a
representative heterogeneous expert pool and must not be used for research
conclusions.

CSV contract checked against the upstream code:

- `expert_inference.py` reads `question`, parses `solvers` with
  `ast.literal_eval`, and uses `gold_answer` for smoke scoring.
- `recruit_agents.py` writes a pandas CSV with a string representation of the
  solver list; this fixture uses that same representation.
