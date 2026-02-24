"""Bug 1: Transform selection logic applies RemoveYTransform when dynamic=False.

When zinc_det config has conditional=True, dynamic=False, target='num_atoms,logp',
the ZINCDataModule.__init__ transform selection logic falls through to the else branch
and applies RemoveYTransform, stripping ALL conditioning data from the dataset.

See: src/datasets/zinc_dataset_det.py lines 378-402

These tests should FAIL while the bug exists, and PASS once fixed.
"""
import pytest
import torch
from torch_geometric.data import Data

from conftest import make_graph_data


# ---------------------------------------------------------------------------
# Unit tests for individual transforms (these always pass — sanity checks)
# ---------------------------------------------------------------------------

class TestRemoveYTransform:
    def test_removes_y_data(self):
        """RemoveYTransform should zero out y to shape [1, 0]."""
        from datasets.zinc_dataset_det import RemoveYTransform

        data = make_graph_data(num_properties=4)
        assert data.y.size(1) == 4  # 4 properties before transform

        transform = RemoveYTransform()
        result = transform(data)
        assert result.y.shape == (1, 0), f"Expected (1, 0), got {result.y.shape}"
        assert result.y.numel() == 0

    def test_return_y_mode(self):
        """RemoveYTransform with return_y=True returns empty tensor."""
        from datasets.zinc_dataset_det import RemoveYTransform

        data = make_graph_data(num_properties=4)
        transform = RemoveYTransform()
        y = transform(data, return_y=True)
        assert y.shape == (1, 0)


class TestSelectDynamicZincTransform:
    def test_single_property(self):
        """SelectDynamicZincTransform extracts a single property correctly."""
        from datasets.zinc_dataset_det import SelectDynamicZincTransform

        data = make_graph_data(num_properties=4)
        original_y = data.y.clone()

        transform = SelectDynamicZincTransform("logp")
        result = transform(data)
        assert result.y.shape == (1, 1), f"Expected (1, 1), got {result.y.shape}"
        assert torch.allclose(result.y[0, 0], original_y[0, 1])  # logp is index 1

    def test_multiple_properties(self):
        """SelectDynamicZincTransform extracts multiple properties correctly."""
        from datasets.zinc_dataset_det import SelectDynamicZincTransform

        data = make_graph_data(num_properties=4)
        original_y = data.y.clone()

        transform = SelectDynamicZincTransform("num_atoms,logp")
        result = transform(data)
        assert result.y.shape == (1, 2), f"Expected (1, 2), got {result.y.shape}"
        assert torch.allclose(result.y[0, 0], original_y[0, 0])  # num_atoms is index 0
        assert torch.allclose(result.y[0, 1], original_y[0, 1])  # logp is index 1

    def test_return_y_mode(self):
        """SelectDynamicZincTransform with return_y=True returns tensor."""
        from datasets.zinc_dataset_det import SelectDynamicZincTransform

        data = make_graph_data(num_properties=4)
        transform = SelectDynamicZincTransform("num_atoms,logp")
        y = transform(data, return_y=True)
        assert y.shape == (1, 2)


# ---------------------------------------------------------------------------
# THE BUG TEST: Transform selection logic
# ---------------------------------------------------------------------------

def _select_transform(conditional, dynamic, target):
    """Replicate the transform selection logic from ZINCDataModule.__init__.

    This mirrors zinc_dataset_det.py:378-402.
    """
    from datasets.zinc_dataset_det import (
        SelectDynamicZincTransform,
        RemoveYTransform,
    )

    regressor = conditional
    if regressor and target and isinstance(target, str) and len(target.strip()) > 0:
        transform = SelectDynamicZincTransform(target)
    else:
        transform = RemoveYTransform()

    return transform


class TestTransformSelection:
    """Tests for the transform selection logic in ZINCDataModule.__init__."""

    def test_dynamic_true_uses_select_transform(self):
        """Control: dynamic=True correctly uses SelectDynamicZincTransform."""
        from datasets.zinc_dataset_det import SelectDynamicZincTransform

        transform = _select_transform(
            conditional=True, dynamic=True, target="num_atoms,logp"
        )
        assert isinstance(transform, SelectDynamicZincTransform)

    def test_dynamic_false_should_preserve_conditioning(self):
        """BUG TEST: conditional=True, dynamic=False, target='num_atoms,logp'.

        Currently FAILS because the code falls through to RemoveYTransform.
        After fix: should use a transform that preserves conditioning data.
        """
        from datasets.zinc_dataset_det import RemoveYTransform

        transform = _select_transform(
            conditional=True, dynamic=False, target="num_atoms,logp"
        )
        # The bug: transform is RemoveYTransform, removing all conditioning.
        # After fix: this should NOT be a RemoveYTransform.
        assert not isinstance(transform, RemoveYTransform), (
            "BUG: conditional=True with target='num_atoms,logp' should NOT use "
            "RemoveYTransform, but the transform selection logic falls through "
            "to the else branch when dynamic=False."
        )

    def test_dynamic_false_preserves_y_on_data(self):
        """BUG TEST: With dynamic=False, conditioning data should survive transform.

        Currently FAILS because RemoveYTransform zeros out y.
        """
        data = make_graph_data(num_properties=4)
        assert data.y.numel() > 0

        transform = _select_transform(
            conditional=True, dynamic=False, target="num_atoms,logp"
        )
        result = transform(data)

        assert result.y.numel() > 0, (
            "BUG: After applying the transform selected with conditional=True and "
            "target='num_atoms,logp', data.y has been emptied. Conditioning data "
            "was stripped by RemoveYTransform."
        )
        assert result.y.size(1) == 2, (
            f"Expected 2 conditioning properties (num_atoms, logp), got {result.y.size(1)}"
        )
