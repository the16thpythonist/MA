#!/bin/bash
# Train DeFoG on zinc_det with conditional generation (num_atoms + logp)
#
# This script trains the model with all conditioning bug fixes applied:
#   - Bug 1: SelectDynamicZincTransform used instead of RemoveYTransform
#   - Bug 2: Unconditional rate matrix uses unconditional predictions
#   - Bug 3: output_dims["y"] derived from data, lambda_train=[5,1], y_loss_type=mse
#
# Requires: DeFoG/.venv/ (aslurmx auto-activates it)
#
# Usage:
#   SLURM:   cd DeFoG && aslurmx -cn haicore_1gpu cmd bash train_zinc_det_conditional.sh
#   Local:   bash train_zinc_det_conditional.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate .venv if not already active (aslurmx does this automatically on SLURM)
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

cd "$SCRIPT_DIR/src"

python main.py \
    +experiment=zinc_det \
    dataset=zinc_det \
    general.wandb=online
