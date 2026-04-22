import torch
from torch import tensor
from src.models.components.nme import NME
from src.models.components.fr import FR

def test_nme_basic():
    metric = NME()
    target = tensor([[[0.5, 0.5], [0.6, 0.6]]])
    preds = tensor([[[0.51, 0.51], [0.61, 0.61]]])
    # Interoccular distance = sqrt((0.5-0.6)^2 + (0.5-0.6)^2) = sqrt(0.01 + 0.01) = sqrt(0.02) = 0.1414
    # Error for point 1 = sqrt(0.01^2 + 0.01^2) / 0.1414 = sqrt(0.0002) / 0.1414 = 0.01414 / 0.1414 = 0.1
    # Error for point 2 = 0.1
    # Mean error = 0.1
    val = metric(preds, target)
    assert torch.allclose(val, tensor(0.1), atol=1e-3)

def test_fr_basic():
    metric = FR(threshold=0.05)
    target = tensor([[[0.5, 0.5], [0.6, 0.6]]])
    preds = tensor([[[0.51, 0.51], [0.61, 0.61]]]) # error 0.1
    val = metric(preds, target)
    assert val == 1.0 # Both points above 0.05 threshold
