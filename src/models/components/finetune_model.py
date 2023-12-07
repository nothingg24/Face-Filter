import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from typing import Optional
from src.models.dlib_module import DLIBLitModule
import torch
from torch import nn
import hydra
from src.models.components.simple_cnn import SimpleCNN

class FinetuneModel(nn.Module):
    def __init__(
            self,
            model: SimpleCNN = SimpleCNN(),
            checkpoint_file: str = "logs/train/runs/2023-12-01_17-58-12/checkpoints/last.ckpt"):
        super().__init__()
        # self.model = DLIBLitModule.load_from_checkpoint(checkpoint_file, net=model)
        self.model = model
        old_weights = list(torch.load(checkpoint_file)['state_dict'].items())
        new_weights = self.model.state_dict()
        i = 0
        for k, _ in new_weights.items():
            new_weights[k] = old_weights[i][1]
            i += 1
        # self.model.load_state_dict(torch.load(checkpoint_file)['state_dict'])
        self.model.load_state_dict(new_weights)
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
         return self.model(x)
            
if __name__ == "__main__":
    model = FinetuneModel()
    print(model)
    print(model(torch.randn(16, 3, 224, 224)).shape)