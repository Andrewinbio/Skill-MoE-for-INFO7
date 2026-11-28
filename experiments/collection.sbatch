#!/bin/bash
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=skillmoe-routed-collection
set -euo pipefail
source /etc/profile.d/modules.sh
module load Singularity/3.5.2-GCC-5.4.0-2.26
ROOT="${SKILL_MOE_ROOT:-$HOME/Skill-MoE-for-INFO7}"
CACHE="${SKILL_MOE_HUB:-/data/project/nlplabs/models/hf/hub}"
IMAGE="${SKILL_MOE_IMAGE:-/data/project/nlplabs/singularity_images/vllm.sif}"
PHASE="${SKILL_MOE_PHASE:?Set SKILL_MOE_PHASE to profiles, smoke or evaluate}"
case "$PHASE" in profiles|smoke|evaluate) ;; *) exit 2 ;; esac
RUNTIME="${SLURM_TMPDIR:-/tmp}/skillmoe-collection-${SLURM_JOB_ID}"
mkdir -p "$RUNTIME"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
singularity exec --nv --cleanenv --bind "$ROOT:/workspace,$CACHE:/opt/model-cache:ro,$RUNTIME:/runtime" "$IMAGE" \
  env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 SLURM_JOB_ID="$SLURM_JOB_ID" \
  HF_HUB_CACHE=/opt/model-cache HF_HOME=/runtime/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  XDG_CACHE_HOME=/runtime/cache XDG_CONFIG_HOME=/runtime/config \
  VLLM_CACHE_ROOT=/runtime/vllm VLLM_CONFIG_ROOT=/runtime/vllm-config VLLM_NO_USAGE_STATS=1 \
  TORCHINDUCTOR_CACHE_DIR=/runtime/torch TRITON_CACHE_DIR=/runtime/triton \
  FLASHINFER_WORKSPACE_BASE=/runtime/flashinfer CUDA_CACHE_PATH=/runtime/cuda \
  python3 /workspace/experiments/collect.py "$PHASE" --profile-rows "${SKILL_MOE_PROFILE_ROWS:-504}"
