from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, tensor
import math
import numpy as np
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _check_same_shape

class NME(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    distances: Tensor
    total: Tensor

    def __init__(self, keypoint_indices: Optional[Sequence[int]]=[36, 45], **kwargs: Any)->None:
        super().__init__(**kwargs)
        self.keypoint_indices = keypoint_indices
        self.add_state("distances", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor)->None:
        _check_same_shape(preds, target)
        interoccular = torch.linalg.norm(target[:, self.keypoint_indices[0], :] - target[:, self.keypoint_indices[1], :], axis=1, keepdims=True) #np
        normalize_factor = torch.tile(interoccular, [1, 2]) #np
        distances = torch.linalg.norm(((preds - target) / normalize_factor[:, None, :]), axis=-1) #np
     #    distances = distances.T
     #    distances = torch.tensor(distances)
        self.distances+=distances.sum()
        self.total+=distances.numel()#len(distances)


    def compute(self)->Tensor:
        return self.distances / self.total
    
