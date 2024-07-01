import warnings
from typing import Union, Iterable, List, Dict, Tuple, Optional, cast

import torch
from torch import Tensor, inf

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_nan_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = True) -> torch.Tensor:
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return False
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        return True
    else:
        return False

