from typing import Any, Optional, Sequence, Union
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from torch import Tensor, tensor
import math
import numpy as np
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _check_same_shape
from src.models.components.nme import NME

class FR(Metric): #Failure Rate
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    count: Tensor
    total: Tensor

    def __init__(self, threshold=0.1, **kwargs: Any)->None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.add_state("count", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor)->None:
        _check_same_shape(preds, target)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nme = NME().to(device)
        preds = preds.to(device)
        target = target.to(device)
        for p, t in zip(preds, target):
            p = p.unsqueeze(0)
            t = t.unsqueeze(0)
            err_i = nme(p, t)
            # print(err_i)
            if err_i > self.threshold:
                self.count += 1
            self.total += 1

    def compute(self)->Tensor:
        return self.count/self.total
    
