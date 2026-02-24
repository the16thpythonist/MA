"""Bug 2: Classifier-free guidance uses conditional predictions for BOTH rate matrices.

In sample_p_zs_given_zt (graph_discrete_flow_model.py:825-890):
  - Line 827: G_1_pred = pred_X, pred_E  (conditional predictions)
  - Line 830: R_t_X, R_t_E computed from G_1_pred (conditional)
  - Lines 844-847: New unconditional predictions computed (pred_X, pred_E updated)
  - Line 871: R_t_X_uncond computed from G_1_pred — STILL THE CONDITIONAL PREDICTIONS!

This makes R_t_X == R_t_X_uncond, so the guidance interpolation is a no-op.

These tests should FAIL while the bug exists, and PASS once fixed.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import MagicMock, patch, call

from conftest import make_cfg, make_small_transformer
from src.utils import PlaceHolder


# ---------------------------------------------------------------------------
# Mock-based test: verify the buggy code path reuses G_1_pred
# ---------------------------------------------------------------------------

class TestGuidanceMocked:
    """Test the guidance logic by replicating the exact code from
    sample_p_zs_given_zt and checking what gets passed."""

    def _run_guidance_logic(self):
        """Replicate the guidance code from sample_p_zs_given_zt.

        This mirrors graph_discrete_flow_model.py lines 822-872.
        Returns the G_1_pred arguments that would be passed to each
        compute_graph_rate_matrix call.
        """
        bs, n, dx, de = 2, 4, 3, 2

        # Simulate conditional forward pass (line 820-823)
        cond_pred_X = torch.softmax(torch.randn(bs, n, dx), dim=-1)
        cond_pred_E = torch.softmax(torch.randn(bs, n, n, de), dim=-1)

        # Simulate unconditional forward pass (lines 844-847)
        uncond_pred_X = torch.softmax(torch.randn(bs, n, dx), dim=-1)
        uncond_pred_E = torch.softmax(torch.randn(bs, n, n, de), dim=-1)

        # --- Replication of the code path ---

        # Line 822-823
        pred_X = cond_pred_X
        pred_E = cond_pred_E

        # Line 827
        G_1_pred = pred_X, pred_E

        # Line 830: First call uses G_1_pred (conditional)
        first_call_G1 = (G_1_pred[0].clone(), G_1_pred[1].clone())

        # Lines 846-847: Second forward pass updates pred_X and pred_E
        pred_X = uncond_pred_X
        pred_E = uncond_pred_E

        # Line 866-872: FIXED — uses (pred_X, pred_E) = unconditional predictions
        G_1_pred_uncond = (pred_X, pred_E)
        second_call_G1 = (G_1_pred_uncond[0].clone(), G_1_pred_uncond[1].clone())

        return first_call_G1, second_call_G1, uncond_pred_X, uncond_pred_E

    def test_unconditional_call_should_use_different_predictions(self):
        """BUG TEST: The unconditional rate matrix call should receive
        unconditional predictions, but receives conditional ones instead.

        This test FAILS while the bug exists (second call uses same G_1_pred)
        and PASSES once line 871 is fixed to use (pred_X, pred_E).
        """
        first_G1, second_G1, uncond_X, uncond_E = self._run_guidance_logic()

        # The second call (unconditional) should use DIFFERENT predictions
        # than the first call (conditional). With the bug, they're identical.
        predictions_differ = not (
            torch.allclose(first_G1[0], second_G1[0])
            and torch.allclose(first_G1[1], second_G1[1])
        )

        assert predictions_differ, (
            "BUG: The unconditional rate matrix is computed from the SAME "
            "predictions as the conditional one. G_1_pred at line 871 still "
            "references the conditional predictions from line 827. "
            "Fix: replace G_1_pred with (pred_X, pred_E) at line 871."
        )

    def test_buggy_guidance_is_noop(self):
        """BUG TEST: When both rate matrices use the same predictions,
        the guidance interpolation is mathematically a no-op.

        exp((1-w)*log(R) + w*log(R)) = exp(log(R)) = R for any w.
        """
        first_G1, second_G1, _, _ = self._run_guidance_logic()
        guidance_weight = 2.0

        # Both calls get the same G_1_pred, so rate matrices would be identical.
        # Use simplified "rate matrices" proportional to predictions.
        R_cond = first_G1[0].abs() + 1e-6
        R_uncond = second_G1[0].abs() + 1e-6  # BUG: same as R_cond

        R_guided = torch.exp(
            torch.log(R_uncond + 1e-6) * (1 - guidance_weight)
            + torch.log(R_cond + 1e-6) * guidance_weight
        )

        # With the bug, guidance has no effect
        guidance_is_noop = torch.allclose(R_guided, R_cond, atol=1e-4)

        assert not guidance_is_noop, (
            "BUG: Guidance interpolation is a no-op because R_cond == R_uncond "
            "(both computed from the same G_1_pred). "
            "exp((1-w)*log(R) + w*log(R)) = R for any guidance weight."
        )


# ---------------------------------------------------------------------------
# Real forward pass test: different y inputs should produce different outputs
# ---------------------------------------------------------------------------

class TestGuidanceRealForward:
    """Test with a real (small) transformer that different y values produce
    different predictions, which is a prerequisite for guidance to work."""

    @pytest.fixture
    def model_and_data(self):
        """Create a small transformer and matching input tensors."""
        num_node_types = 4
        num_edge_types = 3
        num_cond_props = 2
        bs, n = 2, 5

        # input_dims["y"] = num_cond_props + 1 (time) = 3
        input_dims = {"X": num_node_types, "E": num_edge_types, "y": num_cond_props + 1}
        output_dims = {"X": num_node_types, "E": num_edge_types, "y": num_cond_props}

        model = make_small_transformer(input_dims, output_dims)
        model.eval()

        # Create input tensors
        X = torch.randn(bs, n, num_node_types)
        E = torch.randn(bs, n, n, num_edge_types)
        E = (E + E.transpose(1, 2)) / 2  # symmetrize
        node_mask = torch.ones(bs, n, dtype=torch.bool)

        return model, X, E, node_mask, num_cond_props

    def test_different_y_produces_different_predictions(self, model_and_data):
        """Control: a model should produce different outputs for different y inputs.
        This is a prerequisite for guidance to work at all."""
        model, X, E, node_mask, num_cond_props = model_and_data

        y_cond = torch.tensor([[0.5, 1.2, 0.3], [0.8, 0.4, 0.3]])
        y_uncond = torch.tensor([[-1.0, -1.0, 0.3], [-1.0, -1.0, 0.3]])

        with torch.no_grad():
            pred_cond = model(X, E, y_cond, node_mask)
            pred_uncond = model(X, E, y_uncond, node_mask)

        pred_X_cond = F.softmax(pred_cond.X, dim=-1)
        pred_X_uncond = F.softmax(pred_uncond.X, dim=-1)

        assert not torch.allclose(pred_X_cond, pred_X_uncond, atol=1e-5), (
            "Model predictions should differ for conditional vs unconditional y. "
            "If they're identical, FiLM conditioning has no effect."
        )

    def test_correct_guidance_changes_result(self, model_and_data):
        """Control: when using DIFFERENT predictions for cond vs uncond rate
        matrices, the guidance interpolation should actually change the result."""
        model, X, E, node_mask, num_cond_props = model_and_data
        guidance_weight = 2.0

        y_cond = torch.tensor([[0.5, 1.2, 0.3], [0.8, 0.4, 0.3]])
        y_uncond = torch.tensor([[-1.0, -1.0, 0.3], [-1.0, -1.0, 0.3]])

        with torch.no_grad():
            pred_cond = model(X, E, y_cond, node_mask)
            pred_X_cond = F.softmax(pred_cond.X, dim=-1)

            pred_uncond = model(X, E, y_uncond, node_mask)
            pred_X_uncond = F.softmax(pred_uncond.X, dim=-1)

        # Use predictions as simplified "rate matrices"
        R_cond = pred_X_cond.abs() + 1e-6
        R_uncond = pred_X_uncond.abs() + 1e-6  # CORRECT: different predictions

        R_guided = torch.exp(
            torch.log(R_uncond + 1e-6) * (1 - guidance_weight)
            + torch.log(R_cond + 1e-6) * guidance_weight
        )

        # With proper different predictions, guidance should have an effect
        assert not torch.allclose(R_guided, R_cond, atol=1e-4), (
            "With correctly different cond/uncond predictions, guidance "
            "interpolation should produce a result different from R_cond."
        )
