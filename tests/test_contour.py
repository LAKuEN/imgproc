# -*- coding: utf-8 -*-
"""contour.pyのunittest."""
import pytest

import numpy as np

from imgproc import contour


class TestGetFirstItemIndexInTopLayer:
    """get_first_item_index_in_top_layer() のテストケース."""

    hierarchy_a = np.array([[
        [1, -1, -1, -1],
        [2, 0, -1, -1],
        [3, 1, -1, -1],
        [4, 2, -1, -1],
        [-1, 3, -1, -1],
    ]])

    hierarchy_b = np.array([[
        [1, -1, -1, 4],
        [2, 0, -1, 4],
        [3, 1, -1, 4],
        [-1, 2, -1, 4],
        [-1, -1, 0, -1],
    ]])

    test_data = [
        (hierarchy_a, 0),
        (hierarchy_b, 4),
    ]

    @pytest.mark.parametrize("hierarchy, want", test_data)
    def test_normal(self, hierarchy, want):
        """正常系."""
        got = contour.get_first_item_index_in_top_layer(hierarchy)
        assert want == got


class TestGetFirstItemIndexInSameDepth:
    """get_first_item_index_in_same_depth() のテストケース."""

    hierarchy_a = np.array([[
        [1, -1, -1, -1],
        [2, 0, -1, -1],
        [3, 1, -1, -1],
        [4, 2, -1, -1],
        [-1, 3, -1, -1],
    ]])

    hierarchy_b = np.array([[
        [2, 1, -1, -1],
        [0, -1, -1, -1],
        [3, 0, -1, -1],
        [4, 2, -1, -1],
        [-1, 3, -1, -1],
    ]])

    test_data = [
        (hierarchy_a, 0, 0),
        (hierarchy_a, 2, 0),
        (hierarchy_a, 4, 0),
        (hierarchy_b, 0, 1),
        (hierarchy_b, 1, 1),
    ]

    @pytest.mark.parametrize("hierarchy, idx, want", test_data)
    def test_normal(self, hierarchy, idx, want):
        """正常系."""
        got = contour.get_first_item_index_in_same_depth(hierarchy, idx)
        assert want == got


class TestGetItemIndicesInSameDepth:
    """get_item_idices_in_same_depth() のテストケース."""

    hierarchy = np.array([[
        [1, -1, -1, -1],
        [2, 0, -1, -1],
        [3, 1, -1, -1],
        [4, 2, -1, -1],
        [-1, 3, -1, -1],
    ]])

    test_data = [
        (hierarchy, 0, [0, 1, 2, 3, 4]),
        (hierarchy, 2, [0, 1, 2, 3, 4]),
        (hierarchy, 4, [0, 1, 2, 3, 4]),
    ]

    @pytest.mark.parametrize("hierarchy, idx, want", test_data)
    def test_normal(self, hierarchy, idx, want):
        """正常系."""
        got = contour.get_item_indices_in_same_depth(hierarchy, idx)
        assert want == got
