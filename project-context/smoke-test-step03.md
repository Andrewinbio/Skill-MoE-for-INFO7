# Step 03 Cheaha Smoke Test: MedMCQA

Status: **Substitute-model execution and answer-quality smoke tests completed**
on Cheaha, jobs `41597435` and `41597473`, on 2026-11-27 UTC. The exact upstream Qwen2.5 protocol run remains
unverified. Earlier attempts below are retained as history.

## Follow-up answer-quality run and analysis preparation

Job **41597473** completed with exit `0:0` in **4m 18s**, using Qwen3-8B
non-thinking mode and a 4,096-token cap. All **15 expert + 5 aggregator**
responses passed strict explicit-answer parsing and ended with `stop`, not
`length`. Strict vote and aggregator accuracy are both **4/5 (80%)** on the
five development examples. Helpful/harmful counts are both zero. Changing
both mode and cap is not a controlled ablation of either factor.

Local artifacts: `Skill-MoE-for-INFO7/smoke-runs/medmcqa-step03/substitute-41597473/`.
The directory includes raw generation events, `question-records.jsonl`,
`analysis-summary.json`, executed sources, logs, and `RESULT.md`.

Implemented `Skill-MoE-for-INFO7/analysis/prepare_medmcqa.py` and documented
the record schema and commands in `analysis/README.md`. Features contain only
pre-aggregation information. Router confidence is null for manual fixtures.
Strict parsing rejects unfinished thinking, truncation and missing explicit
answers. Repeated expert draws are restored from upstream reverse-numbered
columns before applying earliest-draw tie breaking. Distinct-expert voting
is also recorded. Invalid answers score incorrect; missing outputs are
excluded from paired analysis with coverage reported.

The earlier job `41597435` has also been reanalyzed strictly: expert voting is
40%, aggregation 80%; ten expert outputs have unfinished thinking, leaving
only five valid expert outputs. This differs from the original permissive
parser's 60% expert score and the simpler explicit-answer substring count.
Raw artifacts and their historical report are retained.

Dataset audit: 4,183 unique local test questions, 504 local training rows,
zero exact-normalized duplicate test groups and zero exact-normalized
train/test overlap. Provisional seed-0 split: **2,507 fit / 831 validation /
840 held-out / 5 development-smoke**. Audit source hashes and group assignments
are in `analysis/medmcqa-data/`. Normalization does not detect near-duplicates,
and local hashes do not establish external benchmark provenance.

Primary settings have subsequently been owner-approved and frozen in
`primary-protocol-final.md`: Qwen3 same-model sampled-response scope, all 4,178
non-smoke questions, existing splits, 8 allocated GPU-hours, and 2-percentage-point
accuracy-loss tolerance. Recruitment/router comparisons are omitted in this
approved scope. The resumable primary collector and frozen policy evaluator
remain to be implemented. The five smoke questions are excluded from
experimental splits; calibration is not fit on these development examples.

## Verified batch result (2026-11-27 UTC)

- Slurm: `COMPLETED`, exit `0:0`, elapsed `00:03:52`.
- One A100 80 GB, 8 CPUs, 48 GB host RAM; upstream vLLM execution path.
- User-approved substitute: `Qwen/Qwen3-8B` for experts and aggregator.
- Cached snapshot: `b968826d9c46dd6066d109eabc6255188de91218`.
- Five aligned MedMCQA questions with three Qwen slots each; seed 0.
- Generation cap: 1,024 tokens, now enforced in the isolated vLLM code.
- Validated five CSV rows, three nonempty response columns (15 expert responses),
  five nonempty aggregator responses, and matching reference questions.
- Upstream parser scores: expert majority **3/5 (60%)**, aggregator **4/5 (80%)**.
- **Output-quality limitation:** nine expert responses and one aggregator response
  lack the explicit `answer is (X)` pattern. Inspected examples end mid-reasoning,
  consistent with the short generation cap. The permissive upstream parser still
  extracts letters from these responses. Scores are pipeline diagnostics, not
  reliable benchmark accuracy. A follow-up quality run needs more generation
  budget or an explicitly documented non-thinking configuration.

Artifacts were retrieved locally into:

```text
Skill-MoE-for-INFO7/smoke-runs/medmcqa-step03/substitute-41597435/
  slurm.log
  smoke-test.log
  skills/MedMCQA/round0_seed0.csv
  skills/MedMCQA/fixed_Qwen_round1_seed0_80.0.json
```

The directory also contains the executed isolated Python sources, solver CSV,
and five-question reference JSON. Remote artifacts have the same repository-relative
path under `/home/avdepas2/Skill-MoE-for-INFO7`; the original batch log is
`/home/avdepas2/skillmoe-substitute-41597435.log`.

### Reproducible batch setup and fixes

Saved batch configuration:
`Skill-MoE-for-INFO7/smoke-runs/medmcqa-step03/cheaha-substitute.sbatch`.
Stage the current `run_smoke.sh` at `/home/avdepas2/skillmoe-run_smoke.sh`
before submitting. The batch configuration assumes the existing Cheaha checkout,
dependency wheels and user packages, shared model cache, and vLLM container.
Batch jobs are the preferred execution method going forward.

Fixes used for the successful run:

1. Explicit Hugging Face cache binding and offline cache environment.
2. Job-local writable vLLM, XDG, Torch, Triton, FlashInfer, and CUDA cache paths;
   disabled vLLM usage statistics. This resolves job `41597319`'s read-only `/data`
   initialization failure.
3. Fail-fast shell execution and unique output directory per job.
4. Isolated vLLM V1 engine shutdown compatibility.
5. Enforced generation-token environment setting in isolated `agent.py`.
6. Trimmed the isolated reference JSON to five rows, so aggregator scoring uses
   the correct denominator rather than the full MedMCQA dataset size.

Shell syntax and `git diff --check` passed. No changes were committed or pushed.

## Git checkout verification

The code directory is a real Git checkout:

```text
root: /Users/andre/Documents/UAB/Grad/Fall2026/NLP/Final Project/Skill-MoE-for-INFO7
branch: info762-experiments
origin: https://github.com/Andrewinbio/Skill-MoE-for-INFO7.git
upstream: https://github.com/dinobby/Skill-MoE.git
```

`git ls-remote` reported `b392eb907a02d96cc46936a2051eb3831664f50b` for the
`main` branch of both `origin` and `upstream` at inspection time. The
implementation map's earlier statement that the directory had no `.git`
metadata is superseded by this direct checkout verification.

## skill.zip inspection

No local `skill.zip` or `skills/` directory was found. The upstream README's
Google Drive link was requested directly using both the shared view URL and
the download endpoint. Both returned HTTP 404. Therefore no archive could be
listed or extracted, and no usable MedMCQA solver CSV could be inspected.

The expected upstream CSV is:

```text
skills/MedMCQA/test_samples_with_keywords_and_solvers_seed0.csv
```

It must contain at least `question`, `gold_answer`, and `solvers` columns. That
archive CSV remains unavailable, so its original expert assignments could not
be inspected.

For this continuation, a clearly labeled manual fixture was created instead:

```text
smoke-runs/medmcqa-step03/run/fixture_medmcqa_smoke.csv
smoke-runs/medmcqa-step03/run/fixture_manifest.md
```

The fixture contains the first five real questions from
`test_data/MedMCQA_test.json`, their upstream gold answers, and three manually
assigned slots per question: `['Qwen', 'Qwen', 'Qwen']`. This deliberately
minimizes checkpoint requirements while exercising the upstream three-slot
with-replacement path. It is not a heterogeneous expert experiment.

## Isolated Cheaha job

The job script is:

```text
Skill-MoE-for-INFO7/smoke-runs/medmcqa-step03/run_smoke.sh
```

It uses this separate run directory and leaves the original source/data
untouched:

```text
Skill-MoE-for-INFO7/smoke-runs/medmcqa-step03/run/
```

The script supports these configuration variables:

- `SKILL_MOE_SOLVER_CSV`: path to the pre-generated solver CSV.
- `SKILL_MOE_MODEL_CACHE`: writable model-cache directory; defaults inside the
  isolated run directory.
- `RUN_DIR`: alternate isolated output directory.
- `PROJECT_ROOT`: checkout location.

When inputs are supplied, it copies only the required upstream files and
MedMCQA test data into the run directory, selects the first five aligned CSV
rows, rewrites only those copies to use `SKILL_MOE_MODEL_CACHE`, and invokes
the upstream commands on one Cheaha GPU. The original hard-coded cache path remains
unchanged in the checkout.

The exact commands staged by the script are:

```bash
CUDA_VISIBLE_DEVICES=0 python expert_inference.py --task MedMCQA --gpus 1 --seed 0

CUDA_VISIBLE_DEVICES=0 python aggregate.py --task MedMCQA --aggregator Qwen --gpus 1 --seed 0
```

The aggregator is explicitly `Qwen`, which resolves to
`Qwen/Qwen2.5-7B-Instruct` in the inspected upstream code.

The fixture was validated before any GPU check with Python's standard-library
CSV parser. It has five rows, the required `question`, `gold_answer`, and
`solvers` columns, exact question/gold-answer alignment with the benchmark,
and three `Qwen` slots per row. A pandas-based validation could not run on the
local host because `pandas` is not installed; the standard-library validation
passed. The local inference environment uses `pandas`, `torch`, and
`transformers`; the local backend does not require vLLM.

The public Hugging Face `config.json` endpoint for
`Qwen/Qwen2.5-7B-Instruct` responded successfully from this host. This checks
public metadata access only; it does not verify a local weights cache, available
memory, Hugging Face authentication, or model license/access status.

## Local environment

Created Conda environment:

```bash
conda create -y -n skill-moe-local python=3.10.12
```

Installed successfully from the requirements other than `vllm==0.7.1`:

```bash
conda run -n skill-moe-local python -m pip install \
  tqdm pandas torch==2.5.1 protobuf==5.28.3 sentencepiece==0.2.0 \
  backoff==2.2.1 accelerate==1.0.1 math_verify==0.5.2 \
  sentence_transformers==3.4.1 tiktoken
```

The environment imports `pandas`, `torch`, and `transformers`, and PyTorch
reports MPS availability. `tiktoken` was installed separately because
`utils.py` imports it although it is absent from the checked-in
`requirements.txt`. Installing the complete `requirements.txt` fails at
`vllm==0.7.1`: no compatible macOS ARM wheel is available, and the source
package metadata is reported as `0.7.1+cpu`. The local implementation now uses
the Transformers/MPS backend instead of requiring vLLM. It defaults the Qwen
label to `Qwen/Qwen2.5-0.5B-Instruct` for local memory limits; override
`SKILL_MOE_MODEL_Qwen` to use the upstream 7B checkpoint.

## Attempt and log

The script was first run from the checkout without a fixture and recorded the
missing-CSV blocker. After creating the fixture, it was run with:

```bash
SKILL_MOE_SOLVER_CSV="$PWD/smoke-runs/medmcqa-step03/run/fixture_medmcqa_smoke.csv" \
SKILL_MOE_MODEL_CACHE="$PWD/smoke-runs/medmcqa-step03/run/model-cache" \
bash smoke-runs/medmcqa-step03/run_smoke.sh
```

It created:

```text
smoke-runs/medmcqa-step03/run/smoke-test.log
```

The updated exact blocker recorded in the log is now:

```text
BLOCKED: local upstream inference dependencies are unavailable.
```

The prior local script exited with status `3` before model loading because
vLLM was missing. A separate Conda environment named `skill-moe-local` was
created with Python 3.10.12 and successfully installed the non-vLLM
dependencies, including `pandas 2.3.3`, `torch 2.5.1`, and `transformers
4.57.6`; PyTorch reports MPS available. The local backend now bypasses vLLM
and uses Transformers on MPS, with the local smoke script defaulting to
`Qwen/Qwen2.5-1.5B-Instruct`. No expert or aggregator output has yet been
produced, and no scoring result has been observed.

```text
smoke-runs/medmcqa-step03/run/smoke-test.log
```

## Protocol differences preserved for this smoke test

If a compatible local inference environment becomes available, this smoke test is
intentionally a check of upstream behavior, not the final protocol scoring
implementation:

- It preserves upstream `get_alphabet_choice` parsing and `is_math_equiv`
  scoring.
- Upstream returns `N/A` for an invalid answer, but the current majority code
  can include `N/A` as a vote. The protocol requires invalid answers to count as
  incorrect without silently participating as a valid answer category.
- Upstream uses `Counter(...).most_common(1)` for the vote; the protocol
  requires the documented earliest selected-slot rule for ties.
- Upstream does not log per-question router probabilities, timing, or token
  counts. This smoke test does not claim those protocol fields are available.
- The five-row CSV subset is only a smoke-test reduction. It is not the frozen
  fit/validation/held-out dataset and cannot support calibration or policy
  conclusions.

## Historical Local Attempt

The following local command is retained only as a record of the abandoned
attempt; it is not the active execution path:

```bash
cd /path/to/Skill-MoE-for-INFO7
export SKILL_MOE_SOLVER_CSV="$PWD/smoke-runs/medmcqa-step03/run/fixture_medmcqa_smoke.csv"
export SKILL_MOE_MODEL_CACHE="$PWD/smoke-runs/medmcqa-step03/run/model-cache"
bash smoke-runs/medmcqa-step03/run_smoke.sh
```

The script checks local Python dependencies, copies code and data into the
isolated run directory, validates/aligned-selects five rows, rewrites only
copied scripts to use `SKILL_MOE_MODEL_CACHE`, and then runs the local commands
listed above. It leaves the original checkout data and source unchanged.

## Active Cheaha command

The local path is abandoned. On Cheaha, run the existing isolated job from the
checkout root with the upstream vLLM environment and a writable cache:

```bash
cd /path/to/Skill-MoE-for-INFO7
export SKILL_MOE_SOLVER_CSV="$PWD/smoke-runs/medmcqa-step03/run/fixture_medmcqa_smoke.csv"
export SKILL_MOE_MODEL_CACHE="${SCRATCH:-$PWD}/skill-moe-step03-model-cache"
export SKILL_MOE_BACKEND=vllm
bash smoke-runs/medmcqa-step03/run_smoke.sh
```

The active job requires `vllm`, `pandas`, `torch`, and `transformers`, checks
`nvidia-smi`, uses `CUDA_VISIBLE_DEVICES=0`, and passes
`--aggregator Qwen`. It uses the upstream `Qwen/Qwen2.5-7B-Instruct` mapping;
no local 0.5B override is active on this path.

## Latest Step 03 Attempt

The restored Cheaha job was invoked from this local Mac workspace with the
isolated fixture:

```bash
SKILL_MOE_SOLVER_CSV="$PWD/smoke-runs/medmcqa-step03/run/fixture_medmcqa_smoke.csv" \
SKILL_MOE_MODEL_CACHE="$PWD/smoke-runs/medmcqa-step03/run/model-cache" \
SKILL_MOE_BACKEND=vllm \
bash smoke-runs/medmcqa-step03/run_smoke.sh
```

The job reported `execution_target: cheaha` but was not executed on Cheaha.
This local host lacks `pandas`, `torch`, `transformers`, and `vllm`, so the
preflight stopped before `nvidia-smi`, model loading, or inference. The exact
observed blocker was:

```text
BLOCKED: local upstream inference dependencies are unavailable.
```

The attempt is logged in
`smoke-runs/medmcqa-step03/run/smoke-test.log`; no expert output, aggregator
output, or scoring result was produced.

## Exact Cheaha blockers and decisions

Before Cheaha model inference can run, the project owner must provide or
confirm:

1. Authorized Cheaha access with one GPU and a compatible environment
   containing `vllm`, `pandas`, `torch`, and `transformers`.
2. Access to `Qwen/Qwen2.5-7B-Instruct`, its Hugging Face license/authentication
   requirements, and a writable model cache.
3. Confirmation that the deliberately repeated-Qwen expert fixture is
   sufficient for this smoke test, rather than a heterogeneous pool.
4. Whether the upstream invalid-answer and tie behavior is acceptable for this
   smoke test, while keeping the protocol differences documented above.
