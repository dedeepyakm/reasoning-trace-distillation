#!/bin/bash
# ===========================================================================
# Phoenix job script -- reasoning-trace pilot (Pilot 0/1)
#
#   cd /hpcfs/users/$USER/pilot
#   sbatch phoenix_job.sh
#   squeue -u $USER          # watch it
#   rcstat <jobid>           # stats after it finishes
#
# VERIFY BEFORE FIRST SUBMIT (partition/GPU syntax differs per cluster):
#   sinfo -s                 # list partitions
#   sinfo -o "%P %G"         # which partitions have GPUs, and the gres name
# Then fix the -p line below if 'a100' is not the right partition name.
# ===========================================================================

#SBATCH -p a100                      # TODO verify with `sinfo -s`
#SBATCH -N 1                         # one node
#SBATCH -n 8                         # CPU cores for dataloading
#SBATCH --gres=gpu:1                 # one A100
#SBATCH --time=06:00:00              # ask for what you need; over-asking delays scheduling
#SBATCH --mem=64GB
#SBATCH --job-name=trace-pilot
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dedeepya.korukonda@adelaide.edu.au

# --- CRITICAL: keep model downloads off /home ------------------------------
# /home has a ~10GB quota; gpt-oss-20b alone is ~13GB. Without this the
# download fails partway with a confusing disk-quota error.
export HF_HOME=/hpcfs/users/$USER/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

module purge
module load Anaconda3/2023.03            # or a CUDA-enabled Python module
source activate /hpcfs/users/$USER/envs/pilot   # see setup notes below

nvidia-smi
echo "=== starting $(date) ==="

cd /hpcfs/users/$USER/pilot

# Teacher candidates, full precision on an A100 -- no quantisation caveat here.
python generate_traces.py --model Qwen/Qwen2.5-7B-Instruct \
       --dataset math500 --n 200 --k 4 --max-new-tokens 3072

python generate_traces.py --model openai/gpt-oss-20b \
       --dataset math500 --n 200 --k 4 --max-new-tokens 3072

# Student-side confidence over the same problems (needed for Task 3).
python generate_traces.py --model Qwen/Qwen2.5-1.5B-Instruct \
       --dataset math500 --n 200 --k 4 --max-new-tokens 3072

# Contamination control: GSM8K slice.
python generate_traces.py --model Qwen/Qwen2.5-7B-Instruct \
       --dataset gsm8k --n 50 --k 4 --max-new-tokens 1024

python analyze_pilot.py traces_*.jsonl | tee analysis_report.txt

echo "=== finished $(date) ==="

# ---------------------------------------------------------------------------
# ONE-TIME SETUP on the head node (do this before the first sbatch):
#
#   export HF_HOME=/hpcfs/users/$USER/.cache/huggingface
#   module load Anaconda3/2023.03
#   conda create -p /hpcfs/users/$USER/envs/pilot python=3.11 -y
#   source activate /hpcfs/users/$USER/envs/pilot
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install transformers datasets accelerate scipy pandas
#
# Pre-download weights on the head node (compute nodes may lack internet):
#   python -c "from huggingface_hub import snapshot_download as d; \
#              d('Qwen/Qwen2.5-7B-Instruct'); d('openai/gpt-oss-20b')"
#
# Keep head-node work light -- it is for logins, compiling and short tests only.
# ---------------------------------------------------------------------------
