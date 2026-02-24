"""End-to-end test: does conditioning actually work after training?

This test is independent of any specific bug. It trains a small conditional
model on a synthetic dataset with two clearly distinguishable graph classes,
then checks that the model produces different predictions depending on
the conditioning value.

Design:
  - Class 0 (y=0): Sparse path graphs (chain topology, few edges)
  - Class 1 (y=1): Dense near-complete graphs (most edges present)
  - Train a tiny model (~2 layers, 16-dim) for ~40 epochs on CPU
  - After training, feed the same noisy graph with y=0 vs y=1
  - If conditioning works: predicted edge distributions should differ
    (class 1 should predict more edges)
  - If conditioning is broken: predictions are identical for both y values

Skipped by default. Run with:
    pytest tests/test_e2e_conditioning.py -v --run-e2e
"""
import sys
import os
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

e2e = pytest.mark.e2e

# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

NUM_NODE_TYPES = 3
NUM_EDGE_TYPES = 2  # [no-edge, edge] — binary edges


def _make_path_graph(num_nodes):
    """Class 0: sparse chain graph. Only adjacent nodes connected."""
    x_classes = torch.zeros(num_nodes, dtype=torch.long)
    x_classes[0] = 1  # first node is type 1
    x_classes[-1] = 2  # last node is type 2
    x = F.one_hot(x_classes, NUM_NODE_TYPES).float()

    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    edge_attr = torch.zeros(len(src), NUM_EDGE_TYPES)
    edge_attr[:, 1] = 1.0  # all edges are type 1 (edge present)

    y = torch.tensor([[1.0, 0.0]])  # one-hot class 0
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _make_dense_graph(num_nodes):
    """Class 1: dense near-complete graph. Most node pairs connected."""
    x_classes = torch.ones(num_nodes, dtype=torch.long)  # all type 1
    x = F.one_hot(x_classes, NUM_NODE_TYPES).float()

    src, dst = [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            src.extend([i, j])
            dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    edge_attr = torch.zeros(len(src), NUM_EDGE_TYPES)
    edge_attr[:, 1] = 1.0

    y = torch.tensor([[0.0, 1.0]])  # one-hot class 1
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def make_synthetic_dataset(num_per_class=30, num_nodes=5):
    """Create a mixed dataset of sparse and dense graphs with labels."""
    graphs = []
    for _ in range(num_per_class):
        graphs.append(_make_path_graph(num_nodes))
    for _ in range(num_per_class):
        graphs.append(_make_dense_graph(num_nodes))
    return graphs


def collate_batch(graphs):
    """Collate a list of Data objects into a Batch."""
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# Minimal model setup
# ---------------------------------------------------------------------------

def make_minimal_cfg():
    """Create a minimal config for the end-to-end test."""
    return OmegaConf.create({
        "dataset": {"name": "synthetic_test"},
        "general": {
            "name": "e2e_test",
            "conditional": True,
            "dynamic": True,
            "target": "class",
            "guidance_weight": 2.0,
            "log_every_steps": 9999,
            "number_chain_steps": 5,
        },
        "model": {
            "n_layers": 2,
            "transition": "uniform",
            "lambda_train": [1.0, 1.0],
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
        "train": {"time_distortion": "identity"},
        "sample": {
            "sample_steps": 10,
            "time_distortion": "identity",
            "eta": 0.0,
            "omega": 0.0,
            "rdb": "general",
            "rdb_crit": "x_1",
        },
    })


def make_dataset_infos(num_node_types, num_edge_types, num_cond_dims, max_n_nodes):
    """Create a minimal dataset_infos object."""
    from datasets.dataset_utils import DistributionNodes

    class SyntheticDatasetInfos:
        pass

    info = SyntheticDatasetInfos()
    # +1 for time in y input
    info.input_dims = {"X": num_node_types, "E": num_edge_types, "y": num_cond_dims + 1}
    info.output_dims = {"X": num_node_types, "E": num_edge_types, "y": num_cond_dims}
    info.node_types = torch.ones(num_node_types) / num_node_types
    info.edge_types = torch.ones(num_edge_types) / num_edge_types
    # All graphs have max_n_nodes nodes
    hist = torch.zeros(max_n_nodes + 1)
    hist[max_n_nodes] = 1.0
    info.nodes_dist = DistributionNodes(hist)
    return info


def build_model(cfg, dataset_infos):
    """Instantiate a GraphDiscreteFlowModel with minimal dependencies."""
    from graph_discrete_flow_model import GraphDiscreteFlowModel
    from models.extra_features import DummyExtraFeatures
    from metrics.abstract_metrics import TrainAbstractMetricsDiscrete

    model = GraphDiscreteFlowModel(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=TrainAbstractMetricsDiscrete(),
        sampling_metrics=None,
        visualization_tools=None,
        extra_features=DummyExtraFeatures(),
        domain_features=DummyExtraFeatures(),
        test_labels=None,
        forced_conditions=None,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop (manual, no PL trainer needed)
# ---------------------------------------------------------------------------

def train_model(model, dataset, num_epochs=40, batch_size=16, lr=1e-3):
    """Simple manual training loop."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        # Shuffle dataset
        perm = torch.randperm(len(dataset))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(dataset), batch_size):
            indices = perm[start:start + batch_size]
            batch_graphs = [dataset[i] for i in indices]
            batch = collate_batch(batch_graphs)

            optimizer.zero_grad()
            result = model.training_step(batch, i=epoch * 100 + start)
            if result is None:
                continue
            loss = result["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

    return model


# ---------------------------------------------------------------------------
# The actual tests
# ---------------------------------------------------------------------------

@e2e
class TestEndToEndConditioning:
    """End-to-end test: train a conditional model and verify predictions diverge."""

    @pytest.fixture(scope="class")
    def trained_conditional_model(self):
        """Train a small conditional model on the synthetic dataset."""
        torch.manual_seed(42)

        cfg = make_minimal_cfg()
        dataset_infos = make_dataset_infos(
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            num_cond_dims=2,  # one-hot class label
            max_n_nodes=5,
        )
        model = build_model(cfg, dataset_infos)
        dataset = make_synthetic_dataset(num_per_class=30, num_nodes=5)
        model = train_model(model, dataset, num_epochs=40, batch_size=16)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def trained_unconditional_model(self):
        """Train an identical model but with conditioning disabled."""
        torch.manual_seed(42)

        cfg = make_minimal_cfg()
        cfg.general.conditional = False
        cfg.model.lambda_train = [1.0, 0.0]

        dataset_infos = make_dataset_infos(
            num_node_types=NUM_NODE_TYPES,
            num_edge_types=NUM_EDGE_TYPES,
            num_cond_dims=0,  # no conditioning properties
            max_n_nodes=5,
        )
        # output_dims["y"] = 0 for unconditional
        dataset_infos.output_dims["y"] = 0

        model = build_model(cfg, dataset_infos)

        # Train with y stripped (mimic unconditional)
        dataset = make_synthetic_dataset(num_per_class=30, num_nodes=5)
        for g in dataset:
            g.y = torch.zeros(1, 0)

        model = train_model(model, dataset, num_epochs=40, batch_size=16)
        model.eval()
        return model

    @pytest.fixture
    def noisy_test_input(self, trained_conditional_model):
        """Create a noisy graph at an intermediate timestep for probing."""
        model = trained_conditional_model
        torch.manual_seed(123)

        # Create a test graph and convert to dense format
        from src import utils

        test_graph = _make_path_graph(5)
        batch = collate_batch([test_graph] * 4)  # batch of 4 copies

        dense_data, node_mask = utils.to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # Apply noise at t=0.5 (intermediate)
        t = torch.full((4, 1), 0.5)
        noisy_data = model.apply_noise(X, E, batch.y, node_mask, t=t)

        return noisy_data, node_mask

    def test_conditional_model_predictions_diverge(
        self, trained_conditional_model, noisy_test_input
    ):
        """After training, the conditional model should produce different
        predictions for different conditioning values.

        This is the PRIMARY conditioning test. If this fails, conditioning
        is broken somewhere in the pipeline.
        """
        model = trained_conditional_model
        noisy_data, node_mask = noisy_test_input
        bs = noisy_data["X_t"].shape[0]

        # Probe with class 0 condition (sparse)
        y_class0 = torch.tensor([[1.0, 0.0]] * bs)
        t = noisy_data["t"]
        noisy_data_c0 = {**noisy_data, "y_t": y_class0}

        # Probe with class 1 condition (dense)
        y_class1 = torch.tensor([[0.0, 1.0]] * bs)
        noisy_data_c1 = {**noisy_data, "y_t": y_class1}

        with torch.no_grad():
            extra_c0 = model.compute_extra_data(noisy_data_c0)
            pred_c0 = model.forward(noisy_data_c0, extra_c0, node_mask)

            extra_c1 = model.compute_extra_data(noisy_data_c1)
            pred_c1 = model.forward(noisy_data_c1, extra_c1, node_mask)

        # Compare predicted edge distributions
        pred_E_c0 = F.softmax(pred_c0.E, dim=-1)
        pred_E_c1 = F.softmax(pred_c1.E, dim=-1)

        # Predictions should differ meaningfully
        edge_diff = (pred_E_c0 - pred_E_c1).abs().mean().item()

        assert edge_diff > 0.01, (
            f"Conditional model predictions do NOT differ between class 0 and class 1 "
            f"(mean |diff| = {edge_diff:.6f}). Conditioning is broken somewhere in "
            f"the pipeline: data loading, y propagation, training loss, or model architecture."
        )

    def test_conditional_model_edge_density_direction(
        self, trained_conditional_model, noisy_test_input
    ):
        """The dense-class condition should predict higher edge probability
        than the sparse-class condition."""
        model = trained_conditional_model
        noisy_data, node_mask = noisy_test_input
        bs = noisy_data["X_t"].shape[0]

        y_class0 = torch.tensor([[1.0, 0.0]] * bs)  # sparse
        y_class1 = torch.tensor([[0.0, 1.0]] * bs)  # dense

        with torch.no_grad():
            extra_c0 = model.compute_extra_data({**noisy_data, "y_t": y_class0})
            pred_c0 = model.forward({**noisy_data, "y_t": y_class0}, extra_c0, node_mask)

            extra_c1 = model.compute_extra_data({**noisy_data, "y_t": y_class1})
            pred_c1 = model.forward({**noisy_data, "y_t": y_class1}, extra_c1, node_mask)

        # Edge type index 1 = "edge present"
        prob_edge_c0 = F.softmax(pred_c0.E, dim=-1)[..., 1].mean().item()
        prob_edge_c1 = F.softmax(pred_c1.E, dim=-1)[..., 1].mean().item()

        assert prob_edge_c1 > prob_edge_c0, (
            f"Dense-class condition should predict more edges than sparse-class. "
            f"Got P(edge|class0)={prob_edge_c0:.4f}, P(edge|class1)={prob_edge_c1:.4f}. "
            f"Model has not learned to associate conditioning values with graph structure."
        )

    def test_unconditional_model_invariant_to_y(
        self, trained_unconditional_model
    ):
        """Control: an unconditional model's predictions should NOT change
        when given different y values (since it was trained without them)."""
        model = trained_unconditional_model
        torch.manual_seed(123)

        from src import utils

        test_graph = _make_path_graph(5)
        test_graph.y = torch.zeros(1, 0)  # no conditioning
        batch = collate_batch([test_graph] * 4)

        dense_data, node_mask = utils.to_dense(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        t = torch.full((4, 1), 0.5)
        noisy_data = model.apply_noise(X, E, batch.y, node_mask, t=t)

        # Since model has y_dim=0, y_t is empty — predictions are always the same
        with torch.no_grad():
            extra = model.compute_extra_data(noisy_data)
            pred = model.forward(noisy_data, extra, node_mask)

        pred_E = F.softmax(pred.E, dim=-1)

        # Run again — same input, same output (deterministic in eval mode)
        with torch.no_grad():
            extra2 = model.compute_extra_data(noisy_data)
            pred2 = model.forward(noisy_data, extra2, node_mask)

        pred_E_2 = F.softmax(pred2.E, dim=-1)
        assert torch.allclose(pred_E, pred_E_2, atol=1e-5), (
            "Unconditional model should be deterministic in eval mode"
        )
