import numpy as np
import torch
from .core import maximum_path_jit


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device = neg_cent.device
    dtype = neg_cent.dtype

    neg_cent_np = np.ascontiguousarray(
        neg_cent.detach().cpu().numpy(), dtype=np.float32
    )
    path_np = np.zeros(neg_cent_np.shape, dtype=np.int32)

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].detach().cpu().numpy(), dtype=np.int32
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].detach().cpu().numpy(), dtype=np.int32
    )

    maximum_path_jit(path_np, neg_cent_np, t_t_max, t_s_max)
    return torch.from_numpy(path_np).to(device=device, dtype=dtype)