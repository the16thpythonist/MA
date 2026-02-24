"""Shared fixtures for DeFoG conditioning bug tests."""
import sys
import os
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Data

# Add src to path so DeFoG modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# pytest hooks for custom CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--run-e2e", action="store_true", default=False,
                     help="Run slow end-to-end conditioning tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-e2e"):
        skip = pytest.mark.skip(reason="needs --run-e2e option to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip)


def make_cfg(
    conditional=True,
    dynamic=True,
    target="num_atoms,logp",
    guidance_weight=2.0,
    lambda_train=None,
    n_layers=2,
    sample_steps=10,
):
    """Create a minimal OmegaConf config for testing."""
    if lambda_train is None:
        lambda_train = [5, 0]
    return OmegaConf.create(
        {
            "dataset": {"name": "zinc_det"},
            "general": {
                "name": "test",
                "conditional": conditional,
                "dynamic": dynamic,
                "target": target,
                "guidance_weight": guidance_weight,
            },
            "model": {
                "n_layers": n_layers,
                "transition": "marginal",
                "lambda_train": lambda_train,
                "hidden_mlp_dims": {"X": 32, "E": 16, "y": 16},
                "hidden_dims": {
                    "dx": 16,
                    "de": 8,
                    "dy": 8,
                    "n_head": 2,
                    "dim_ffX": 32,
                    "dim_ffE": 16,
                },
            },
            "sample": {
                "sample_steps": sample_steps,
                "eta": 0.0,
                "omega": 0.0,
                "rdb": "general",
                "rdb_crit": "dummy",
                "time_distortion": "identity",
            },
            "train": {
                "time_distortion": "identity",
            },
        }
    )


def make_graph_data(num_nodes=5, num_node_types=4, num_edge_types=3, num_properties=2):
    """Create a fake PyG Data object with valid structure.

    Returns a graph with one-hot node/edge features and float property values in y.
    """
    # One-hot node features
    node_classes = torch.randint(0, num_node_types, (num_nodes,))
    x = torch.zeros(num_nodes, num_node_types)
    x.scatter_(1, node_classes.unsqueeze(1), 1.0)

    # Simple chain graph: 0-1-2-...-n
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # One-hot edge features (skip index 0 = no-edge, so offset by 1)
    num_edges = edge_index.size(1)
    edge_classes = torch.randint(1, num_edge_types, (num_edges,))
    edge_attr = torch.zeros(num_edges, num_edge_types)
    edge_attr.scatter_(1, edge_classes.unsqueeze(1), 1.0)

    # Property values (e.g. num_atoms=5.0, logp=1.2, ...)
    y = torch.randn(1, num_properties).abs()

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        idx=0,
    )


def make_small_transformer(input_dims, output_dims):
    """Create a minimal 2-layer GraphTransformer for testing."""
    from models.transformer_model import GraphTransformer

    return GraphTransformer(
        n_layers=2,
        input_dims=input_dims,
        hidden_mlp_dims={"X": 32, "E": 16, "y": 16},
        hidden_dims={
            "dx": 16,
            "de": 8,
            "dy": 8,
            "n_head": 2,
            "dim_ffX": 32,
            "dim_ffE": 16,
        },
        output_dims=output_dims,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    )
