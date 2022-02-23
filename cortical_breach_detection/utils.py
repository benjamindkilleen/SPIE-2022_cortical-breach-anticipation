#!/usr/bin/env python3
"""Contains generic util functions needed for your project."""
import json
import logging
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from deepdrr import geo

log = logging.getLogger(__name__)


def doublewrap(f):
    """
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    """

    @wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


def jsonable(obj: Any):
    """Convert obj to a JSON-ready container or object.

    Args:
        obj ([type]):
    """
    if isinstance(obj, (str, float, int, complex)):
        return obj
    elif isinstance(obj, Path):
        return str(obj.resolve())
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map(jsonable, obj))
    elif isinstance(obj, dict):
        return dict(jsonable(list(obj.items())))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "__array__"):
        return np.array(obj).tolist()
    else:
        raise ValueError(f"Unknown type for JSON: {type(obj)}")


def save_json(path: str, obj: Any):
    obj = jsonable(obj)
    with open(path, "w") as file:
        json.dump(obj, file)


def load_json(path: str):
    with open(path, "r") as file:
        out = json.load(file)
    return out


def split_sizes(size: int, split: np.ndarray) -> np.ndarray:
    """Get a non-random split into a dataset with `size` elements.

    The minimum size for each split is 1. If size is not large enough to accommodate this, then an
    error is raised.

    Returns:
        np.ndarray: The size of each split, so that the `i`th section corresponds to
            `data[split[i]:split[i+1]]`.

    """
    split = np.array(split)
    assert np.all(split >= 0)
    assert np.sum(split) == 1, f"invalid split: {split}"
    assert size >= np.sum(split > 0)

    sizes = np.floor(size * split)
    for i in range(sizes.shape[0]):
        if sizes[i] > 0 or split[i] == 0:
            continue

        idx = np.argmax(split)
        sizes[i] += 1
        sizes[idx] -= 1

    if np.sum(sizes) != size:
        idx = np.argmax(sizes)
        sizes[idx] += size - np.sum(sizes)

    assert np.sum(sizes) == size, f"split sizes {sizes.tolist()} does not sum to {size}"
    sizes = sizes.astype(np.int64)
    return sizes


def split_indices(size: int, split: np.ndarray) -> np.ndarray:
    """Get start indices of a non-random split into a dataset with `size` elements.

    The minimum size for each split is 1. If size is not large enough to accommodate this, then an
    error is raised.

    Returns:
        np.ndarray: The start index of each split, so that the `i`th section corresponds to
            `data[split[i]:split[i+1]]`. Includes the trailing index for convenience.

    """
    sizes = split_sizes(size, split)
    indices = [0]
    for s in sizes:
        indices.append(indices[-1] + s)

    assert indices[-1] == size
    return np.array(indices)


def heatmap(x: Tuple[float, float], size: Tuple[int, int], sigma: float = 3) -> torch.Tensor:
    mu_y, mu_x = x
    H, W = size
    xs, ys = torch.meshgrid(
        torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32)
    )
    den = 2 * sigma * sigma
    heatmap = torch.exp(-((xs - mu_x).pow(2) + (ys - mu_y).pow(2)) / den)
    heatmap /= heatmap.max()
    heatmap = heatmap.unsqueeze(0)
    return heatmap
