#!/usr/bin/env bash
set -euo pipefail

# Run the upstream scoring path on Cheaha in an isolated directory. Set
# SKILL_MOE_SOLVER_CSV to the fixture or another MedMCQA solver CSV before
# running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/run}"
MODEL_CACHE="${SKILL_MOE_MODEL_CACHE:-$RUN_DIR/model-cache}"
SOLVER_CSV="${SKILL_MOE_SOLVER_CSV:-}"
MODEL_OVERRIDE="${SKILL_MOE_MODEL_QWEN:-}"
LOG_FILE="$RUN_DIR/smoke-test.log"

mkdir -p "$RUN_DIR" "$MODEL_CACHE"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'Step 03 MedMCQA smoke test\n'
printf 'timestamp: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'project_root: %s\nrun_dir: %s\nmodel_cache: %s\n' "$PROJECT_ROOT" "$RUN_DIR" "$MODEL_CACHE"
printf 'execution_target: cheaha\n'
printf 'origin: '
git -C "$PROJECT_ROOT" remote get-url origin 2>/dev/null || true
printf 'upstream: '
git -C "$PROJECT_ROOT" remote get-url upstream 2>/dev/null || true

if [[ -z "$SOLVER_CSV" || ! -f "$SOLVER_CSV" ]]; then
    printf '\nBLOCKED: no usable pre-generated MedMCQA solver CSV was supplied.\n'
    printf 'Expected a file containing question, gold_answer, and solvers columns.\n'
    printf 'Set SKILL_MOE_SOLVER_CSV=/path/to/test_samples_with_keywords_and_solvers_seed0.csv\n'
    printf 'The advertised upstream skill.zip URL returned HTTP 404 during Step 03.\n'
    exit 2
fi

export SKILL_MOE_BACKEND="${SKILL_MOE_BACKEND:-vllm}"
export SKILL_MOE_MAX_NEW_TOKENS="${SKILL_MOE_MAX_NEW_TOKENS:-4096}"
printf 'backend: %s\nmodel_override: %s\nmax_new_tokens: %s\n' \
    "$SKILL_MOE_BACKEND" "${MODEL_OVERRIDE:-upstream agent_map}" "$SKILL_MOE_MAX_NEW_TOKENS"

if ! python - <<'PY'
required = ("pandas", "torch", "transformers", "vllm")
missing = []
for name in required:
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    print("missing Python packages: " + ", ".join(missing))
    raise SystemExit(1)
print("local Python dependencies are available")
PY
then
    printf '\nBLOCKED: local upstream inference dependencies are unavailable.\n'
    printf 'Install the upstream Cheaha dependencies before model inference.\n'
    exit 3
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '\nBLOCKED: nvidia-smi is unavailable; this host is not verified as Cheaha GPU runtime.\n'
    exit 3
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$RUN_DIR/skills/MedMCQA" "$RUN_DIR/test_data"
cp "$PROJECT_ROOT/agent.py" "$PROJECT_ROOT/utils.py" "$PROJECT_ROOT/local_backend.py" \
    "$PROJECT_ROOT/expert_inference.py" "$PROJECT_ROOT/aggregate.py" "$RUN_DIR/"
cp "$PROJECT_ROOT/test_data/MedMCQA_test.json" "$RUN_DIR/test_data/"

if [[ -n "$MODEL_OVERRIDE" ]]; then
    perl -0pi -e 's#Qwen/Qwen2\.5-7B-Instruct#$ENV{SKILL_MOE_MODEL_QWEN}#g' \
        "$RUN_DIR/agent.py"
fi

# Make only the isolated copies honor the configurable cache. The original
# upstream files remain unchanged.
perl -0pi -e 's#download_dir\s*=\s*"/nas-ssd2/cychen/saved_models"#download_dir=os.environ["SKILL_MOE_MODEL_CACHE"]#g' \
    "$RUN_DIR/expert_inference.py" "$RUN_DIR/aggregate.py"
export SKILL_MOE_MODEL_CACHE="$MODEL_CACHE"

# Compatibility with vLLM V1; only isolated smoke copies are adapted.
python - "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

run = Path(sys.argv[1])
expert = run / "expert_inference.py"
text = expert.read_text()
text = text.replace(
    "    del llm.llm_engine.model_executor",
    '    if hasattr(llm.llm_engine, "model_executor"):\n'
    '        del llm.llm_engine.model_executor\n'
    '    elif hasattr(llm.llm_engine, "engine_core"):\n'
    '        llm.llm_engine.engine_core.shutdown()',
)
expert.write_text(text)
agent = run / "agent.py"
text = agent.read_text().replace(
    "max_tokens=max_tokens)",
    'max_tokens=int(os.environ.get("SKILL_MOE_MAX_NEW_TOKENS", max_tokens)))',
)
text = text.replace(
    "add_generation_prompt=True",
    'enable_thinking=os.environ.get("SKILL_MOE_ENABLE_THINKING", "1") == "1", add_generation_prompt=True',
)
text = text.replace(
    "    outputs = llm.generate(messages, sampling_params)",
    '''    import time
    import json
    started = time.perf_counter()
    outputs = llm.generate(messages, sampling_params)
    elapsed = time.perf_counter() - started
    with open("generation-events.jsonl", "a") as handle:
        for index, output in enumerate(outputs):
            completion = output.outputs[0]
            handle.write(json.dumps({
                "stage": os.environ["SKILL_MOE_STAGE"], "index": index,
                "model": agent_map[agent_name], "agent": agent_name,
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "batch_generation_seconds": elapsed,
                "raw_output": completion.text,
                "max_tokens": sampling_params.max_tokens,
                "enable_thinking": os.environ.get("SKILL_MOE_ENABLE_THINKING", "1") == "1",
            }) + "\\n")''',
)
agent.write_text(text)
PY

# Keep the first five rows so the input remains a deterministic 3-5 question
# smoke test. The upstream scripts themselves have no --limit option.
python - "$SOLVER_CSV" "$RUN_DIR/skills/MedMCQA/test_samples_with_keywords_and_solvers_seed0.csv" "$PROJECT_ROOT/test_data/MedMCQA_test.json" <<'PY'
import ast
import json
import sys
from pathlib import Path

import pandas as pd

source, target, reference = sys.argv[1:]
frame = pd.read_csv(source)
required = {"question", "gold_answer", "solvers"}
missing = required.difference(frame.columns)
if missing:
    raise SystemExit(f"solver CSV is missing columns: {sorted(missing)}")
if len(frame) < 3:
    raise SystemExit("solver CSV contains fewer than three rows")

subset = frame.iloc[:5].copy()
with open(reference) as handle:
    benchmark = json.load(handle)
for index, row in subset.iterrows():
    if row["question"] != benchmark[index]["question"]:
        raise SystemExit(f"solver CSV row {index} does not align with MedMCQA test data")
    if not isinstance(ast.literal_eval(row["solvers"]), list):
        raise SystemExit(f"solver CSV row {index} has a non-list solvers value")

subset.to_csv(target, index=False)
# Match the isolated evaluation denominator to the five-row fixture.
with (Path(target).parents[2] / "test_data/MedMCQA_test.json").open("w") as handle:
    json.dump(benchmark[:len(subset)], handle)
print(f"prepared_rows: {len(subset)}")
print("solver_rows:")
for index, row in subset.iterrows():
    print(f"  {index}: {row['solvers']}")
PY

cd "$RUN_DIR"
printf '\nCOMMAND: CUDA_VISIBLE_DEVICES=0 python expert_inference.py --task MedMCQA --gpus 1 --seed 0\n'
SKILL_MOE_STAGE=expert CUDA_VISIBLE_DEVICES=0 python expert_inference.py --task MedMCQA --gpus 1 --seed 0

printf '\nCOMMAND: CUDA_VISIBLE_DEVICES=0 python aggregate.py --task MedMCQA --aggregator Qwen --gpus 1 --seed 0\n'
SKILL_MOE_STAGE=aggregator CUDA_VISIBLE_DEVICES=0 python aggregate.py --task MedMCQA --aggregator Qwen --gpus 1 --seed 0

printf '\nsmoke test completed\n'
