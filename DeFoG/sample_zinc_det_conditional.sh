#!/bin/bash
# Sample from trained DeFoG zinc_det model with property conditioning
#
# Generates 1000 molecules conditioned on fixed (num_atoms, logP) targets,
# computes achieved properties, and saves comparison plots + CSV.
#
# This script submits a SLURM job on HAICORE via aslurmx.
# Run from the DeFoG repo root (where .venv is located).
#
# Edit target values directly in experiments/sample_conditional.py
#
# Requires: DeFoG/.venv/ (aslurmx auto-activates it)

set -euo pipefail

cd "$(dirname "$0")"

aslurmx -cn haicore_1gpu -o time=01:00:00 \
    cmd python experiments/sample_conditional.py
