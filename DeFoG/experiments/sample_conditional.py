"""
Conditional Sampling Experiment for DeFoG (ZINC_det)

Generates molecules conditioned on fixed target properties (num_atoms, logP),
then compares the achieved property distributions against the training dataset
to demonstrate that conditioning is working.

Usage:
    cd DeFoG && python experiments/sample_conditional.py
"""

import os
import sys
import math
import random
import string
import warnings
from datetime import datetime

# ── Hardcoded defaults (edit these) ──────────────────────────────────────────
DEFOG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(
    DEFOG_ROOT,
    "..",
    "outputs/2026-02-25/15-21-42-a23n-zinc_det-zinc_det/checkpoints/zinc_det/epoch=95.ckpt",
)
CONFIG_PATH = os.path.join(
    DEFOG_ROOT,
    "..",
    "outputs/2026-02-25/15-21-42-a23n-zinc_det-zinc_det/.hydra/config.yaml",
)
TARGET_NUM_ATOMS = 20
TARGET_LOGP = 4.0
NUM_SAMPLES = 1000
BATCH_SIZE = 256
GUIDANCE_WEIGHT = 3.0
OUTPUTS_ROOT = os.path.join(DEFOG_ROOT, "..", "outputs")
# ─────────────────────────────────────────────────────────────────────────────

# Add DeFoG/src to path so we can import project modules
sys.path.insert(0, os.path.join(DEFOG_ROOT, "src"))

os.environ["TORCH_WEIGHTS_ONLY"] = "0"

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode
from collections import defaultdict
from typing import Any
import typing

# Torch safe globals for checkpoint loading
torch.serialization.add_safe_globals([
    DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
    np.dtype, type(np.array(0).item()),
    dict, list, int, defaultdict, Any,
    torch.distributions.categorical.Categorical,
])
for _name in ("_SpecialForm", "_GenericAlias", "_UnionGenericAlias", "_SpecialGenericAlias"):
    if hasattr(typing, _name):
        torch.serialization.add_safe_globals([getattr(typing, _name)])

# Project imports (require src on sys.path)
from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from graph_discrete_flow_model import GraphDiscreteFlowModel
from models.extra_features import ExtraFeatures
from models.extra_features_molecular import ExtraMolecularFeatures
from metrics.molecular_metrics import SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from analysis.visualization import MolecularVisualization
from analysis.rdkit_functions import build_molecule, compute_logp
from datasets import zinc_dataset_det

warnings.filterwarnings("ignore")


def create_output_dir():
    """Create an output directory following the DeFoG naming convention:
    outputs/<date>/<HH-MM-SS>-<4char_id>-conditional_sampling-zinc_det/
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    folder_name = f"{time_str}-{rand_id}-conditional_sampling-zinc_det"
    output_dir = os.path.join(OUTPUTS_ROOT, date_str, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_sampling_config(output_dir):
    """Save the sampling parameters as a YAML file for reproducibility."""
    sampling_cfg = OmegaConf.create({
        "checkpoint": os.path.realpath(CHECKPOINT),
        "training_config": os.path.realpath(CONFIG_PATH),
        "target_num_atoms": TARGET_NUM_ATOMS,
        "target_logp": TARGET_LOGP,
        "num_samples": NUM_SAMPLES,
        "batch_size": BATCH_SIZE,
        "guidance_weight": GUIDANCE_WEIGHT,
    })
    path = os.path.join(output_dir, "sampling_config.yaml")
    OmegaConf.save(sampling_cfg, path)
    print(f"Config saved to: {path}")


def save_generated_csv(output_dir, achieved_logp, achieved_num_atoms, smiles_list):
    """Save generated molecules and their properties as CSV."""
    df = pd.DataFrame({
        "smiles": smiles_list,
        "num_atoms": achieved_num_atoms,
        "logp": achieved_logp,
    })
    path = os.path.join(output_dir, "generated_molecules.csv")
    df.to_csv(path, index=False)
    print(f"CSV saved to: {path} ({len(df)} molecules)")


def save_summary(output_dir, valid_count, total, achieved_logp, achieved_num_atoms):
    """Save a text summary of the results."""
    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("CONDITIONAL SAMPLING RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Checkpoint:      {os.path.realpath(CHECKPOINT)}\n")
        f.write(f"Target:          num_atoms={TARGET_NUM_ATOMS}, logP={TARGET_LOGP}\n")
        f.write(f"Guidance weight: {GUIDANCE_WEIGHT}\n")
        f.write(f"Generated:       {total} molecules\n")
        f.write(f"Valid:            {valid_count} ({100 * valid_count / total:.1f}%)\n")
        if achieved_logp:
            mae_logp = np.mean(np.abs(np.array(achieved_logp) - TARGET_LOGP))
            mae_atoms = np.mean(np.abs(np.array(achieved_num_atoms) - TARGET_NUM_ATOMS))
            f.write(f"logP MAE:        {mae_logp:.3f}\n")
            f.write(f"num_atoms MAE:   {mae_atoms:.3f}\n")
            f.write(f"logP mean:       {np.mean(achieved_logp):.3f}  (target: {TARGET_LOGP})\n")
            f.write(f"logP std:        {np.std(achieved_logp):.3f}\n")
            f.write(f"num_atoms mean:  {np.mean(achieved_num_atoms):.1f}  (target: {TARGET_NUM_ATOMS})\n")
            f.write(f"num_atoms std:   {np.std(achieved_num_atoms):.1f}\n")
        f.write("=" * 60 + "\n")
    print(f"Summary saved to: {path}")


def load_model(cfg, forced_conditions):
    """Bootstrap all model components and load checkpoint weights."""
    # Data module
    datamodule = zinc_dataset_det.ZINCDataModule(cfg)
    dataset_infos = zinc_dataset_det.ZINCinfos(datamodule=datamodule, cfg=cfg)
    dataset_smiles = zinc_dataset_det.get_smiles(
        cfg=cfg, datamodule=datamodule, dataset_infos=dataset_infos, evaluate_datasets=False
    )

    extra_features = ExtraFeatures(
        cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=dataset_infos
    )
    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features, domain_features=domain_features
    )

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    add_virtual_states = "absorbing" == cfg.model.transition
    sampling_metrics = SamplingMolecularMetrics(
        dataset_infos, dataset_smiles, cfg, add_virtual_states=add_virtual_states
    )
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    dataset_infos.compute_reference_metrics(
        datamodule=datamodule, sampling_metrics=sampling_metrics
    )

    model = GraphDiscreteFlowModel(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=train_metrics,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization_tools,
        extra_features=extra_features,
        domain_features=domain_features,
        test_labels=datamodule.test_labels if cfg.general.conditional else None,
        forced_conditions=forced_conditions,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT}")
    raw_ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
    elif isinstance(raw_ckpt, dict):
        state_dict = raw_ckpt
    else:
        raise ValueError(f"Unknown checkpoint format: {type(raw_ckpt)}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    print("Checkpoint loaded successfully.")

    return model, dataset_infos


def generate_molecules(model, num_samples, batch_size):
    """Generate molecules in batches using the model's sample_batch."""
    device = next(model.parameters()).device
    all_molecules = []
    all_labels = []
    num_batches = math.ceil(num_samples / batch_size)

    for i in range(num_batches):
        bs = min(batch_size, num_samples - len(all_molecules))
        print(f"  Batch {i + 1}/{num_batches}: generating {bs} molecules...")
        molecules, labels = model.sample_batch(
            batch_id=i * batch_size,
            batch_size=bs,
            keep_chain=0,
            number_chain_steps=1,
            save_final=0,
            save_visualization=False,
        )
        all_molecules.extend(molecules)
        all_labels.extend(labels)

    return all_molecules, all_labels


def compute_generated_properties(molecules, atom_decoder):
    """Compute logP, num_atoms, and SMILES for each generated molecule."""
    from rdkit import Chem
    from analysis.rdkit_functions import mol2smiles

    achieved_logp = []
    achieved_num_atoms = []
    smiles_list = []
    valid_count = 0
    total = len(molecules)

    for atom_types, edge_types in molecules:
        mol = build_molecule(atom_types, edge_types, atom_decoder)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        valid_count += 1
        achieved_logp.append(compute_logp(mol))
        achieved_num_atoms.append(mol.GetNumAtoms())
        smiles_list.append(mol2smiles(mol) or "")

    return achieved_logp, achieved_num_atoms, smiles_list, valid_count, total


def create_plots(
    dataset_logp, dataset_num_atoms,
    achieved_logp, achieved_num_atoms,
    target_logp, target_num_atoms,
    output_path,
):
    """Create a 4-panel figure comparing dataset vs generated distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Conditional Generation: target num_atoms={target_num_atoms}, logP={target_logp}",
        fontsize=14,
        fontweight="bold",
    )

    # ── Panel 1: logP histogram overlay ──────────────────────────────────
    ax = axes[0, 0]
    ax.hist(dataset_logp, bins=60, density=True, alpha=0.5, color="steelblue", label="Dataset (ZINC250k)")
    ax.hist(achieved_logp, bins=30, density=True, alpha=0.7, color="tomato", label="Generated")
    ax.axvline(target_logp, color="black", linestyle="--", linewidth=2, label=f"Target = {target_logp}")
    ax.set_xlabel("logP")
    ax.set_ylabel("Density")
    ax.set_title("logP Distribution")
    ax.legend()

    # ── Panel 2: num_atoms histogram overlay ─────────────────────────────
    ax = axes[0, 1]
    ds_min = int(min(dataset_num_atoms))
    ds_max = int(max(dataset_num_atoms))
    bins_atoms = np.arange(ds_min - 0.5, ds_max + 1.5, 1)
    ax.hist(dataset_num_atoms, bins=bins_atoms, density=True, alpha=0.5, color="steelblue", label="Dataset (ZINC250k)")
    gen_min = int(min(achieved_num_atoms)) if achieved_num_atoms else ds_min
    gen_max = int(max(achieved_num_atoms)) if achieved_num_atoms else ds_max
    bins_gen = np.arange(min(ds_min, gen_min) - 0.5, max(ds_max, gen_max) + 1.5, 1)
    ax.hist(achieved_num_atoms, bins=bins_gen, density=True, alpha=0.7, color="tomato", label="Generated")
    ax.axvline(target_num_atoms, color="black", linestyle="--", linewidth=2, label=f"Target = {target_num_atoms}")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Density")
    ax.set_title("Number of Atoms Distribution")
    ax.legend()

    # ── Panel 3: Achieved logP strip plot ────────────────────────────────
    ax = axes[1, 0]
    jitter = np.random.normal(0, 0.04, size=len(achieved_logp))
    ax.scatter(
        jitter, achieved_logp, alpha=0.4, s=12, color="tomato", edgecolors="none"
    )
    ax.axhline(target_logp, color="black", linestyle="--", linewidth=2, label=f"Target = {target_logp}")
    mae_logp = np.mean(np.abs(np.array(achieved_logp) - target_logp))
    ax.set_ylabel("Achieved logP")
    ax.set_xticks([])
    ax.set_title(f"Achieved logP per Molecule  (MAE = {mae_logp:.3f})")
    ax.legend()

    # ── Panel 4: Achieved num_atoms strip plot ───────────────────────────
    ax = axes[1, 1]
    jitter = np.random.normal(0, 0.04, size=len(achieved_num_atoms))
    ax.scatter(
        jitter, achieved_num_atoms, alpha=0.4, s=12, color="tomato", edgecolors="none"
    )
    ax.axhline(target_num_atoms, color="black", linestyle="--", linewidth=2, label=f"Target = {target_num_atoms}")
    mae_atoms = np.mean(np.abs(np.array(achieved_num_atoms) - target_num_atoms))
    ax.set_ylabel("Achieved Number of Atoms")
    ax.set_xticks([])
    ax.set_title(f"Achieved num_atoms per Molecule  (MAE = {mae_atoms:.3f})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def save_molecule_grid(smiles_list, output_path, n=25):
    """Save an RDKit grid image of n randomly chosen valid molecules."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    valid_smiles = [s for s in smiles_list if s]
    if not valid_smiles:
        print("No valid SMILES to visualize.")
        return

    chosen = random.sample(valid_smiles, min(n, len(valid_smiles)))
    mols = [Chem.MolFromSmiles(s) for s in chosen]
    mols = [m for m in mols if m is not None]

    if not mols:
        print("No valid RDKit mols for grid image.")
        return

    n_per_row = 5
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_per_row,
        subImgSize=(300, 300),
        legends=[f"{i+1}" for i in range(len(mols))],
    )
    img.save(output_path)
    print(f"Molecule grid ({len(mols)} mols) saved to: {output_path}")


def main():
    # Create output directory: outputs/<date>/<time>-<id>-conditional_sampling-zinc_det/
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")

    # Load config
    print(f"Loading config from: {CONFIG_PATH}")
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.general.guidance_weight = GUIDANCE_WEIGHT

    # Forced condition vector: [num_atoms, logp] matching the target order
    forced_conditions = torch.tensor([TARGET_NUM_ATOMS, TARGET_LOGP], dtype=torch.float)
    print(f"Target conditions: num_atoms={TARGET_NUM_ATOMS}, logP={TARGET_LOGP}")
    print(f"Guidance weight: {GUIDANCE_WEIGHT}")

    # Save sampling config for reproducibility
    save_sampling_config(output_dir)

    # Build model and load checkpoint
    model, dataset_infos = load_model(cfg, forced_conditions)

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model on device: {device}")

    # Generate molecules
    print(f"\nGenerating {NUM_SAMPLES} molecules...")
    molecules, labels = generate_molecules(model, NUM_SAMPLES, BATCH_SIZE)

    # Compute properties on generated molecules
    print("\nComputing properties on generated molecules...")
    achieved_logp, achieved_num_atoms, smiles_list, valid_count, total = compute_generated_properties(
        molecules, dataset_infos.atom_decoder
    )

    # Load dataset distribution from CSV
    csv_path = os.path.join(DEFOG_ROOT, "data", "zinc_det", "zinc_250k_rdkit.csv")
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    dataset_logp = df["logp"].values
    dataset_num_atoms = df["num_atoms"].values

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Target:          num_atoms={TARGET_NUM_ATOMS}, logP={TARGET_LOGP}")
    print(f"  Guidance weight: {GUIDANCE_WEIGHT}")
    print(f"  Generated:       {total} molecules")
    print(f"  Valid:            {valid_count} ({100 * valid_count / total:.1f}%)")
    if achieved_logp:
        mae_logp = np.mean(np.abs(np.array(achieved_logp) - TARGET_LOGP))
        mae_atoms = np.mean(np.abs(np.array(achieved_num_atoms) - TARGET_NUM_ATOMS))
        print(f"  logP MAE:        {mae_logp:.3f}")
        print(f"  num_atoms MAE:   {mae_atoms:.3f}")
        print(f"  logP mean:       {np.mean(achieved_logp):.3f}  (target: {TARGET_LOGP})")
        print(f"  num_atoms mean:  {np.mean(achieved_num_atoms):.1f}  (target: {TARGET_NUM_ATOMS})")
    print("=" * 60)

    # Save all artifacts to output directory
    plot_path = os.path.join(output_dir, "conditional_sampling_results.png")
    create_plots(
        dataset_logp, dataset_num_atoms,
        achieved_logp, achieved_num_atoms,
        TARGET_LOGP, TARGET_NUM_ATOMS,
        plot_path,
    )
    save_generated_csv(output_dir, achieved_logp, achieved_num_atoms, smiles_list)
    save_summary(output_dir, valid_count, total, achieved_logp, achieved_num_atoms)
    save_molecule_grid(
        smiles_list,
        os.path.join(output_dir, "sampled_molecules_grid.png"),
        n=25,
    )

    print(f"\nAll artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
