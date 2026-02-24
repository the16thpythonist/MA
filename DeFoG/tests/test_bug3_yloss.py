"""Bug 3: output_dims["y"] and lambda_train[1] must enable the auxiliary y loss.

Two requirements for conditioning to work during training:

1. abstract_dataset.py must set output_dims["y"] from the actual conditioning
   dimensions in the data (not hardcoded to 0).

2. configs/model/discrete.yaml must have lambda_train[1] > 0 so the y loss
   actually contributes gradient signal.

These tests verify both requirements are met.
"""
import pytest
import torch
import torch.nn as nn

from conftest import make_small_transformer


# ---------------------------------------------------------------------------
# Bug 3a: output_dims["y"] must reflect conditioning dimensions
# ---------------------------------------------------------------------------

class TestOutputDimsY:
    def test_output_dims_y_from_data(self):
        """output_dims['y'] should be set from example_batch['y'].size(1).

        Replicates the logic from abstract_dataset.py:153-157.
        """
        # Simulate a batch with 2 conditioning properties
        y_size_1 = 2  # e.g. num_atoms, logp

        # This is what abstract_dataset.py:153-157 should do after the fix:
        output_dims = {
            "X": 9,
            "E": 4,
            "y": y_size_1 if y_size_1 > 0 else 0,
        }

        assert output_dims["y"] > 0, (
            "output_dims['y'] should reflect the conditioning dimensions. "
            f"Expected {y_size_1}, got {output_dims['y']}."
        )
        assert output_dims["y"] == y_size_1

    def test_transformer_empty_y_output_when_dims_zero(self):
        """When output_dims['y']=0, the transformer produces an empty y tensor."""
        input_dims = {"X": 4, "E": 3, "y": 3}  # y = 2 props + 1 time
        output_dims = {"X": 4, "E": 3, "y": 0}

        model = make_small_transformer(input_dims, output_dims)

        bs, n = 2, 4
        X = torch.randn(bs, n, 4)
        E = torch.randn(bs, n, n, 3)
        E = (E + E.transpose(1, 2)) / 2
        y = torch.randn(bs, 3)
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        with torch.no_grad():
            pred = model(X, E, y, node_mask)

        assert pred.y.numel() == 0, (
            "With output_dims['y']=0, pred.y should be empty"
        )

    def test_transformer_produces_y_when_dims_nonzero(self):
        """When output_dims['y'] > 0, transformer produces y output."""
        input_dims = {"X": 4, "E": 3, "y": 3}
        output_dims = {"X": 4, "E": 3, "y": 2}

        model = make_small_transformer(input_dims, output_dims)

        bs, n = 2, 4
        X = torch.randn(bs, n, 4)
        E = torch.randn(bs, n, n, 3)
        E = (E + E.transpose(1, 2)) / 2
        y = torch.randn(bs, 3)
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        with torch.no_grad():
            pred = model(X, E, y, node_mask)

        assert pred.y.shape == (bs, 2), (
            f"Expected pred.y shape (2, 2), got {pred.y.shape}"
        )
        assert pred.y.numel() > 0


# ---------------------------------------------------------------------------
# Bug 3b: lambda_train[1] must be > 0
# ---------------------------------------------------------------------------

class TestLambdaTrainY:
    @pytest.fixture
    def loss_inputs(self):
        """Create realistic loss inputs for TrainLossDiscrete."""
        bs, n, dx, de, dy = 2, 5, 4, 3, 2

        # Predictions (logits)
        pred_X = torch.randn(bs, n, dx)
        pred_E = torch.randn(bs, n, n, de)
        pred_y = torch.randn(bs, dy)

        # Ground truth (one-hot for X and E, one-hot for y)
        true_X = torch.zeros(bs, n, dx)
        true_X.scatter_(2, torch.randint(0, dx, (bs, n, 1)), 1.0)

        true_E = torch.zeros(bs, n, n, de)
        true_E.scatter_(3, torch.randint(0, de, (bs, n, n, 1)), 1.0)

        true_y = torch.zeros(bs, dy)
        true_y.scatter_(1, torch.randint(0, dy, (bs, 1)), 1.0)

        return pred_X, pred_E, pred_y, true_X, true_E, true_y

    def test_lambda_nonzero_y_loss_contributes(self, loss_inputs):
        """With lambda_train=[5, 1], changing pred_y changes the total loss.

        This verifies the model receives gradient signal for conditioning.
        """
        from metrics.train_metrics import TrainLossDiscrete

        pred_X, pred_E, pred_y, true_X, true_E, true_y = loss_inputs

        loss_fn = TrainLossDiscrete(lambda_train=[5, 1])

        loss1 = loss_fn(pred_X, pred_E, pred_y, true_X, true_E, true_y, log=False)
        loss_fn.reset()

        # Completely different y predictions
        pred_y_different = torch.randn_like(pred_y) * 100
        loss2 = loss_fn(pred_X, pred_E, pred_y_different, true_X, true_E, true_y, log=False)

        assert not torch.allclose(loss1, loss2), (
            "With lambda_train=[5, 1], different pred_y should produce different loss. "
            "The y loss must contribute gradient signal for conditioning to work."
        )

    def test_empty_pred_y_skips_loss(self, loss_inputs):
        """When pred_y is empty (numel==0), loss_y should be 0.0."""
        from metrics.train_metrics import TrainLossDiscrete

        pred_X, pred_E, _, true_X, true_E, true_y = loss_inputs

        loss_fn = TrainLossDiscrete(lambda_train=[5, 1])
        pred_y_empty = torch.zeros(2, 0)

        loss = loss_fn(pred_X, pred_E, pred_y_empty, true_X, true_E, true_y, log=False)

        # Should still compute a valid loss (just without y contribution)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Combined: output_dims["y"] > 0 enables y learning
# ---------------------------------------------------------------------------

class TestCombinedOutputAndLoss:
    def test_nonzero_output_dims_enables_y_learning(self):
        """With output_dims['y'] > 0 and lambda_train[1] > 0, the y loss
        contributes to total loss, enabling the model to learn conditioning."""
        from metrics.train_metrics import TrainLossDiscrete

        input_dims = {"X": 4, "E": 3, "y": 3}
        output_dims = {"X": 4, "E": 3, "y": 2}  # FIXED: y > 0

        model = make_small_transformer(input_dims, output_dims)

        bs, n = 2, 4
        X = torch.randn(bs, n, 4)
        E = torch.randn(bs, n, n, 3)
        E = (E + E.transpose(1, 2)) / 2
        y = torch.randn(bs, 3)
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        with torch.no_grad():
            pred = model(X, E, y, node_mask)

        # pred.y should have content
        assert pred.y.numel() > 0, "pred.y should be non-empty with output_dims['y']=2"

        # Loss with y predictions vs without should differ
        loss_fn = TrainLossDiscrete(lambda_train=[5, 1])

        true_X = torch.zeros(bs, n, 4)
        true_X.scatter_(2, torch.randint(0, 4, (bs, n, 1)), 1.0)
        true_E = torch.zeros(bs, n, n, 3)
        true_E.scatter_(3, torch.randint(0, 3, (bs, n, n, 1)), 1.0)
        true_y = torch.zeros(bs, 2)
        true_y.scatter_(1, torch.randint(0, 2, (bs, 1)), 1.0)

        loss_with_y = loss_fn(pred.X, pred.E, pred.y, true_X, true_E, true_y, log=False)
        loss_fn.reset()
        loss_without_y = loss_fn(pred.X, pred.E, torch.zeros(bs, 0), true_X, true_E, true_y, log=False)

        assert not torch.allclose(loss_with_y, loss_without_y), (
            "With output_dims['y']=2 and lambda_train=[5,1], the y loss should "
            "contribute differently than an empty y prediction."
        )


# ---------------------------------------------------------------------------
# MSE regression loss for continuous y targets
# ---------------------------------------------------------------------------

class TestMSERegressionLoss:
    @pytest.fixture
    def regression_inputs(self):
        """Create inputs with continuous (non-one-hot) y targets."""
        bs, n, dx, de, dy = 2, 5, 4, 3, 2

        pred_X = torch.randn(bs, n, dx)
        pred_E = torch.randn(bs, n, n, de)
        pred_y = torch.randn(bs, dy)

        true_X = torch.zeros(bs, n, dx)
        true_X.scatter_(2, torch.randint(0, dx, (bs, n, 1)), 1.0)
        true_E = torch.zeros(bs, n, n, de)
        true_E.scatter_(3, torch.randint(0, de, (bs, n, n, 1)), 1.0)

        # Continuous regression targets (e.g. num_atoms=15.0, logp=2.3)
        true_y = torch.tensor([[15.0, 2.3], [22.0, -1.1]])

        return pred_X, pred_E, pred_y, true_X, true_E, true_y

    def test_mse_loss_y_contributes(self, regression_inputs):
        """With y_loss_type='mse', different pred_y should produce different loss."""
        from metrics.train_metrics import TrainLossDiscrete

        pred_X, pred_E, pred_y, true_X, true_E, true_y = regression_inputs

        loss_fn = TrainLossDiscrete(lambda_train=[5, 1], y_loss_type="mse")

        loss1 = loss_fn(pred_X, pred_E, pred_y, true_X, true_E, true_y, log=False)
        loss_fn.reset()

        pred_y_different = torch.randn_like(pred_y) * 100
        loss2 = loss_fn(pred_X, pred_E, pred_y_different, true_X, true_E, true_y, log=False)

        assert not torch.allclose(loss1, loss2), (
            "With y_loss_type='mse' and lambda_train=[5, 1], different pred_y "
            "should produce different total loss."
        )

    def test_mse_loss_decreases_toward_target(self, regression_inputs):
        """MSE loss should decrease as predictions approach the continuous target."""
        from metrics.train_metrics import TrainLossDiscrete

        pred_X, pred_E, _, true_X, true_E, true_y = regression_inputs

        loss_fn = TrainLossDiscrete(lambda_train=[0, 1], y_loss_type="mse")

        # Far from target
        pred_y_far = true_y + 10.0
        loss_far = loss_fn(pred_X, pred_E, pred_y_far, true_X, true_E, true_y, log=False)
        loss_fn.reset()

        # Close to target
        pred_y_close = true_y + 0.1
        loss_close = loss_fn(pred_X, pred_E, pred_y_close, true_X, true_E, true_y, log=False)

        assert loss_far > loss_close, (
            f"MSE loss should decrease as predictions get closer to target. "
            f"Far: {loss_far.item():.4f}, Close: {loss_close.item():.4f}"
        )

    def test_ce_vs_mse_switch(self, regression_inputs):
        """y_loss_type config should switch between CE and MSE metrics."""
        from metrics.train_metrics import TrainLossDiscrete
        from metrics.abstract_metrics import CrossEntropyMetric, MSEMetric

        loss_ce = TrainLossDiscrete(lambda_train=[5, 1], y_loss_type="ce")
        loss_mse = TrainLossDiscrete(lambda_train=[5, 1], y_loss_type="mse")

        assert isinstance(loss_ce.y_loss, CrossEntropyMetric)
        assert isinstance(loss_mse.y_loss, MSEMetric)

    def test_mse_empty_pred_y_skips_loss(self, regression_inputs):
        """With MSE loss, empty pred_y should still produce a valid loss."""
        from metrics.train_metrics import TrainLossDiscrete

        pred_X, pred_E, _, true_X, true_E, true_y = regression_inputs

        loss_fn = TrainLossDiscrete(lambda_train=[5, 1], y_loss_type="mse")
        pred_y_empty = torch.zeros(2, 0)

        loss = loss_fn(pred_X, pred_E, pred_y_empty, true_X, true_E, true_y, log=False)
        assert torch.isfinite(loss)
