from typing import Tuple
from typing import Union

import numpy as np
import torch


def random_uniform_tensor(
    t: Union[int, float, Tuple[float, float]], dtype=torch.float32
) -> torch.Tensor:
    """Convert t to a tensor, if it is a single number, otherwise sample from the range.

    If the output type is

    Ags:
        t (Union[int, float, Tuple[float, float]]): Either a scalar or a tuple of scalars, (low, high), inclusive.
        dtype: What torch type to return.

    Returns:
        torch.Tensor: A scalar tensor.
    """
    if isinstance(t, tuple):
        lo, hi = t
        t = lo + (hi - lo) * torch.rand(1, dtype=torch.float32)[0]
        return t.to(dtype)
    else:
        return torch.tensor(t, dtype=dtype)
