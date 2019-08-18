# -*- coding: utf-8 -*-
"""輪郭情報を処理する関数モジュール.

* 輪郭情報
* 階層情報
"""
import numpy as np


NEXT_INDEX, PREV_INDEX, FIRST_CHILD_INDEX, PARENT_INDEX = range(4)


def get_first_item_index_in_top_layer(hierarchy: [np.ndarray]) -> int:
    """最上位の輪郭のindexを取得.

    cv2.RETR_TREEで取得した輪郭情報を入力として得る前提
    最も上位層の先頭の輪郭のindexを検索して返す
    """
    for i, h in enumerate(hierarchy[0]):
        if h[PREV_INDEX] == -1 and h[PARENT_INDEX] == -1:
            return i


def get_first_item_index_in_same_depth(hierarchy: [np.ndarray],
                                       idx: int) -> int:
    """同じ階層の内、先頭の要素のindexを返却."""
    prev_idx = hierarchy[0][idx][PREV_INDEX]
    if prev_idx != -1:
        return get_first_item_index_in_same_depth(hierarchy, prev_idx)

    return idx


def get_item_indices_which_connnected_after(hierarchy: [np.ndarray],
                                            idx: int,
                                            idxes: [int] = None) -> int:
    """指定した要素に連なる要素を全て取得."""
    idxes = [] if idxes is None else idxes
    idxes.append(idx)
    next_idx = hierarchy[0][idx][NEXT_INDEX]
    if next_idx != -1:
        idxes = get_item_indices_which_connnected_after(hierarchy,
                                                        next_idx, idxes)

    return idxes


def get_item_indices_in_same_depth(hierarchy: [np.ndarray], idx: int) -> [int]:
    """同じ階層の要素のindexを全件取得し返却."""
    first_idx = get_first_item_index_in_same_depth(hierarchy, idx)
    idxes = get_item_indices_which_connnected_after(hierarchy, first_idx)

    return idxes
