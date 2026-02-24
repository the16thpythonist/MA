#!/bin/bash
# Train DeFoG on zinc_det with conditional generation (num_atoms + logp)
#
# This script submits a SLURM job on HAICORE via aslurmx.
# Run from the DeFoG repo root (where .venv is located).
#
# Conditioning bug fixes applied:
#   - Bug 1: SelectDynamicZincTransform used instead of RemoveYTransform
#   - Bug 2: Unconditional rate matrix uses unconditional predictions
#   - Bug 3: output_dims["y"] derived from data, lambda_train=[5,1], y_loss_type=mse
#
# Requires: DeFoG/.venv/ (aslurmx auto-activates it)

set -euo pipefail

cd "$(dirname "$0")"

aslurmx -cn haicore_1gpu \
    cmd bash -c 'cd src && python main.py +experiment=zinc_det dataset=zinc_det general.wandb=disabled'
