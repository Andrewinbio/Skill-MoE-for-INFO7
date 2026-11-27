# Step 02 Implementation Map: Skill-MoE Upstream

Status: inspected locally and against the public upstream repository. No
inference was run in Step 02.

## Revision and local-copy verification

The public upstream repository is:

- Repository: `https://github.com/dinobby/Skill-MoE`
- Branch inspected: `main`
- Commit: `b392eb907a02d96cc46936a2051eb3831664f50b`
- Commit message: `Update title and citation`
- Commit date reported by GitHub: 2026-05-22

The local `Skill-MoE-for-INFO7` source and MedMCQA data match the Git blob
identifiers reported by that commit for the files checked. This is a content
match, not local Git history: the copied directory has no `.git` metadata.
Checked files include `agent.py`, `recruit_agents.py`, `expert_inference.py`,
`aggregate.py`, `create_profile.py`, `annotate_keywords.py`, `bench_aggr.py`,
`utils.py`, the shell scripts, `README.md`, `requirements.txt`, and
`test_data/MedMCQA_{train,test}.json`.

The revision should be recorded as the pinned upstream revision for subsequent
work. The local copy itself cannot prove when or from which commit it was
created beyond these matching blobs.

## MedMCQA data and paths

The local data files are:

- `Skill-MoE-for-INFO7/test_data/MedMCQA_train.json`
- `Skill-MoE-for-INFO7/test_data/MedMCQA_test.json`

The code resolves these paths relative to the current working directory, not
relative to the Python file. Commands therefore must be run from
`Skill-MoE-for-INFO7/`.

The public upstream MedMCQA blobs at the inspected commit are:

- `MedMCQA_train.json`: Git blob `d4f36a14a277debb4c3fe9f89c3d1ca2f0b1e61f`
- `MedMCQA_test.json`: Git blob `f0e0acd66d8c9520c3b68005dd6cdce04ba5b8b0`

The current files contain the upstream train/test records. Stable question IDs
are not present; records currently contain `question` and `gold_answer` (and
some datasets contain `category`). A later dataset step must create IDs and
freeze the fit/validation/held-out partitions.

## Entry points and execution order

The README describes this order:

1. `annotate_keywords.py` creates keyword annotations under
   `./skills/MedMCQA/` using fixed annotator agent `Qwen`.
2. `create_profile.py` is run once per expert to create profile and chain-of-
   thought files under `./skills/MedMCQA/`.
3. `bench_aggr.py` optionally creates and evaluates synthetic aggregator
   benchmark files.
4. `recruit_agents.py --task MedMCQA --seed 0` maps test keywords to training
   keywords, reads profiles, and writes
   `./skills/MedMCQA/test_samples_with_keywords_and_solvers_seed0.csv`.
5. `expert_inference.py --task MedMCQA --gpus 1 --seed 0` reads that CSV,
   generates expert responses, and writes
   `./skills/MedMCQA/round0_seed0.csv`.
6. `aggregate.py --task MedMCQA --aggregator Qwen --gpus 1 --seed 0` reads the
   round-zero CSV, generates aggregator responses, scores them, and writes a
   `fixed_Qwen_round1_seed0_<accuracy>.json` output.

The README's documented example uses GPQA. Replacing the task with `MedMCQA`
and the aggregator with `Qwen` follows the README's aggregator table, but the
repository does not provide a dedicated MedMCQA shell script.

## Three-slot support

The upstream code supports the proposed three-slot configuration:

- `recruit_agents.py:78-94` defines `recruit_agents(..., k=3)` and calls
  `random.choices(..., k=k)`, so selection is with replacement.
- `recruit_agents.py:187-203` calls that function with `k=3` for each test
  question and writes the resulting model names in the `solvers` column.
- `expert_inference.py:94-136` iterates over every solver occurrence and
  stores responses in per-model columns such as
  `<solver>_answer_1`, `<solver>_answer_2`, and `<solver>_answer_3`.
- The majority calculation in `expert_inference.py:178-192` collects all
  valid answer columns, so repeated selections remain separate vote slots.

Therefore the proposed three slots are supported, including repeated model
selections. The current code does not implement the protocol's deterministic
earliest-draw tie break; it uses `Counter(...).most_common(1)` in first-seen
order. That behavior should be verified or replaced during instrumentation.

## Router and selection behavior

The upstream code does not expose a per-question router probability vector.
The implemented selection path is:

1. `recruit_agents.py:158-162` collects training keywords.
2. `recruit_agents.py:171-186` loads all profile JSON files and computes each
   model's global strength from the sum of profile values.
3. `recruit_agents.py:45-57` converts global strengths to normalized values
   with a temperature of `0.5`.
4. `recruit_agents.py:187-203` sums profile scores for the test question's
   mapped keywords, multiplies positive scores by global strength, normalizes
   the resulting weights, and samples three solver names with replacement.

The normalized weights are used as sampling weights but are not saved. The
code also does not save the per-question weights, draw probabilities, or the
random draw values. Consequently, the protocol's router-confidence predictor
cannot be recovered from the current outputs without adding instrumentation.
The existing routing implementation is stochastic recruitment, not a logged
router-probability API.

## Expert profiles and model checkpoints

### Profile files

For MedMCQA, `recruit_agents.py:145-177` expects:

```text
Skill-MoE-for-INFO7/skills/MedMCQA/train_samples_with_keywords_Qwen.json
Skill-MoE-for-INFO7/skills/MedMCQA/test_samples_with_keywords_Qwen.json
Skill-MoE-for-INFO7/skills/MedMCQA/*_profile_*.json
```

Profiles are produced by `create_profile.py:54-57` from the annotated training
file. The local checkout has no `skills/` directory and therefore contains no
MedMCQA profiles, keyword annotations, solver CSV, cached expert outputs, or
aggregator outputs. Profile existence and profile accuracy cannot be verified
locally.

The checked-in batch script lists this 16-model profile set, although its task
is hard-coded to GPQA rather than MedMCQA:

```text
Llama LlamaR1 Qwen Mistral Phi Gemma GLM Exaone Granite QwenMath QwenCode
DeepSeekMath QwenR1 InternLM Mathstral BioLlama
```

`agent.py:14-32` maps those names to these Hugging Face checkpoints:

```text
Llama       meta-llama/Meta-Llama-3.1-8B-Instruct
LlamaR1     deepseek-ai/DeepSeek-R1-Distill-Llama-8B
Qwen        Qwen/Qwen2.5-7B-Instruct
Mistral     mistralai/Mistral-Nemo-Instruct-2407
Phi         microsoft/Phi-3.5-mini-instruct
Gemma       google/gemma-2-9b-it
GLM         THUDM/glm-4-9b-chat
Exaone      LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
Granite     ibm-granite/granite-3.1-8b-instruct
QwenMath    Qwen/Qwen2.5-Math-7B
QwenCode    Qwen/Qwen2.5-Coder-7B-Instruct
DeepSeekMath deepseek-ai/deepseek-math-7b-instruct
QwenR1      deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
InternLM    internlm/internlm3-8b-instruct
Mathstral   mistralai/Mathstral-7B-v0.1
BioLlama    ContactDoctor/Bio-Medical-Llama-3-8B
```

`Qwen72B` and `Llama70B` are also present in `agent.py`, but are not in the
checked-in profile-generation batch script. The code does not prove that any
of these checkpoints are cached on Cheaha.

### Checkpoint storage and access

The inference and profile scripts hard-code the vLLM download directory:

```text
/nas-ssd2/cychen/saved_models
```

This path is developer-specific and is not verifiable from the local machine.
`AutoTokenizer.from_pretrained` and vLLM also require Hugging Face access for
models that are not already cached. Llama and some other checkpoints may
require an accepted license or authentication. The exact Cheaha cache contents,
Hugging Face credentials, and license access are unverified.

## Exact MedMCQA aggregator

The public upstream README lists `Qwen` as the MedMCQA aggregator. The local
code resolves that name in `agent.py:16` to:

```text
Qwen/Qwen2.5-7B-Instruct
```

`aggregate.py:52-74` loads that checkpoint and calls `vllm_generate`; the
command-line default is instead `QwenR1`, so `--aggregator Qwen` must be
specified for the provisional protocol. The precise Hugging Face revision
(commit or tag) is not pinned by the code or README, so only the model ID is
verified. The local cache and access status remain unverified.

## Answer parsing and scoring

- `expert_inference.py:183-190` parses MedMCQA outputs with
  `get_alphabet_choice(..., num_choice=4)` and scores the modal answer with
  `is_math_equiv`.
- `aggregate.py:88` sets four answer choices for MedMCQA, and lines 145-153
  parses and scores aggregator outputs with the same alphabet parser.
- `utils.py:30-40` first searches parenthesized capital letters and otherwise
  searches any allowed capital letter, returning the last match or `N/A`.
- `utils.py:64-74` uses `math_verify` for scoring, including the answer-choice
  strings.

The parser can extract a letter from reasoning text when the requested final
format is absent, and invalid answers become `N/A`. The current majority code
does not explicitly exclude `N/A`, so invalid outputs can participate as a
vote. This differs from the provisional protocol's invalid-answer rule and
must be resolved in later instrumentation/scoring work.

## Smallest feasible one-GPU smoke command

The smallest documented one-GPU commands for the existing pipeline, run from
`Skill-MoE-for-INFO7/`, are:

```bash
CUDA_VISIBLE_DEVICES=0 python expert_inference.py \
  --task MedMCQA --gpus 1 --seed 0

CUDA_VISIBLE_DEVICES=0 python aggregate.py \
  --task MedMCQA --aggregator Qwen --gpus 1 --seed 0
```

These commands are feasible only after the required `skills/MedMCQA/` inputs
exist, and the current scripts process the full selected CSV rather than a
few examples. The repository has no `--limit` option. A true few-question
smoke test therefore requires either a pre-generated small solver CSV/output
package or a later, explicitly scoped CLI/configuration change. Running
`recruit_agents.py` from this checkout is not currently feasible because the
required keyword annotation and profile files are absent.

The upstream README advertises a Google Drive `skill.zip` containing
pre-generated outputs. Its availability, contents, integrity, and Cheaha
download/access status were not verified. If it contains the expected
MedMCQA files, extraction followed by the two commands above is the smallest
unmodified-path smoke test. The scripts still use the hard-coded model cache
path and may load multiple expert checkpoints serially on the one GPU.

## Findings and unverified items

Verified:

- Current public upstream `main` commit and matching local source/data blobs.
- MedMCQA train/test paths.
- Three solver slots with replacement.
- The 16-model profile-generation list in the checked-in batch script.
- Model IDs in `agent.py`, including the exact `Qwen` aggregator model ID.
- One-GPU CLI arguments for expert inference and aggregation.

Not verified:

- A local upstream Git commit or provenance beyond content-hash matching.
- Any MedMCQA keyword files, expert profiles, cached checkpoints, or generated
  outputs.
- The exact Hugging Face checkpoint revision, Cheaha GPU-hour availability,
  model licenses, authentication, and `/nas-ssd2/cychen/saved_models` access.
- Whether the Google Drive archive is available and complete.
- A successful Cheaha execution or even a few-example run; no GPU execution
  was performed here.
- Per-question router probabilities, because upstream does not save them.

## Decisions still required

Before running the Step 03 smoke test, the project owner must decide or
confirm:

1. Pin the upstream commit above and the exact Hugging Face revision for every
   expert and for `Qwen/Qwen2.5-7B-Instruct`.
2. Confirm the actual MedMCQA expert pool and which profile files are
   authorized; the 16-model shell list is not proof that MedMCQA profiles
   exist.
3. Confirm access to Cheaha, the one-GPU allocation, model cache/checkpoint
   storage, and any Hugging Face licenses or tokens.
4. Decide whether to use the unmodified full-CSV command or approve a minimal
   `--limit`/small-input change for the proposed few-question smoke test.
5. Approve the upstream answer parsing behavior or require the protocol's
   explicit invalid-answer exclusion and deterministic tie rule before any
   scored run.
6. Confirm whether the missing router probabilities are acceptable for Step 03
   or reserve instrumentation as a prerequisite for the planned router-
   confidence analysis.
7. Confirm that the hard-coded download path may be used on Cheaha; otherwise
   approve replacing it with a configurable repository-relative or environment
   variable path in a later reproducibility step.
