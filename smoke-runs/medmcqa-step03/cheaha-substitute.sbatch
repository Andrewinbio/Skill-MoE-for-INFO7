#!/bin/bash
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=skillmoe-substitute
#SBATCH --output=/home/avdepas2/skillmoe-substitute-%j.log
# Stage run_smoke.sh at /home/avdepas2/skillmoe-run_smoke.sh before submitting.
set -euo pipefail
source /etc/profile.d/modules.sh
module load Singularity/3.5.2-GCC-5.4.0-2.26
TMP=/tmp/skillmoe-smoke-${SLURM_JOB_ID}
mkdir -p "$TMP/python"
for pattern in 'pandas-2.2.3-cp312-*.whl' 'math_verify-0.5.2-*.whl' 'latex2sympy2_extended-1.0.6-*.whl' 'antlr4_python3_runtime-4.13.2-*.whl' 'sympy-1.14.0-*.whl' 'mpmath-1.3.0-*.whl' 'backoff-2.2.1-*.whl'; do
    for wheel in /home/avdepas2/python-wheels/$pattern; do
        unzip -q -o "$wheel" -d "$TMP/python"
    done
done
perl -pe 's/\bpython\b/python3/g' /home/avdepas2/skillmoe-run_smoke.sh > "$TMP/smoke.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/opt/local-packages:/opt/python-packages
export HF_HUB_CACHE=/opt/model-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$TMP/torchinductor" TRITON_CACHE_DIR="$TMP/triton"
export XDG_CACHE_HOME="$TMP/cache" XDG_CONFIG_HOME="$TMP/config"
export VLLM_CACHE_ROOT="$TMP/vllm-cache" VLLM_CONFIG_ROOT="$TMP/vllm-config"
export FLASHINFER_WORKSPACE_BASE="$TMP/flashinfer" CUDA_CACHE_PATH="$TMP/cuda"
export VLLM_NO_USAGE_STATS=1 HF_HOME="$TMP/hf"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$CUDA_CACHE_PATH"
export OMP_NUM_THREADS=8
export PROJECT_ROOT=/workspace/Skill-MoE-for-INFO7
export RUN_DIR="$PROJECT_ROOT/smoke-runs/medmcqa-step03/substitute-$SLURM_JOB_ID"
export SKILL_MOE_SOLVER_CSV="$PROJECT_ROOT/smoke-runs/medmcqa-step03/run/fixture_medmcqa_smoke.csv"
export SKILL_MOE_MODEL_CACHE=/opt/model-cache SKILL_MOE_MODEL_QWEN=Qwen/Qwen3-8B
export SKILL_MOE_MAX_NEW_TOKENS="${SKILL_MOE_MAX_NEW_TOKENS:-4096}" SKILL_MOE_BACKEND=vllm
export SKILL_MOE_ENABLE_THINKING="${SKILL_MOE_ENABLE_THINKING:-0}"
printf 'Substitute-model smoke test; job %s; not a Qwen2.5 protocol result\n' "$SLURM_JOB_ID"
singularity exec --nv --bind "/home/avdepas2/Skill-MoE-for-INFO7:$PROJECT_ROOT,/home/avdepas2/python-packages:/opt/python-packages,$TMP/python:/opt/local-packages,$TMP/smoke.sh:/opt/smoke.sh,/data/project/nlplabs/models/hf/hub:/opt/model-cache" /data/project/nlplabs/singularity_images/vllm.sif bash /opt/smoke.sh
