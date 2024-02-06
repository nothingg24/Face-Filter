import torch
from torch import nn
import torchvision
from torchvision.models import list_models, get_model, get_weight

class Swin(nn.Module):
    def __init__(
            self,
            model_name: str = "swin_v2_t",
            weights: str = "DEFAULT",
            output_shape: list = [68, 2]):
        super().__init__()

        backbone = get_model(name=model_name, weights=None)
        
        self.backbone = backbone
        
        self.mlp = nn.Sequential(nn.Linear(1000, 768),
                                 nn.GELU(),
                                 nn.Linear(768, 512),
                                 nn.GELU(),
                                 nn.Linear(512, 256),
                                 nn.GELU(),
                                 nn.Linear(256, output_shape[0] * output_shape[1])
                                 )
        
        self.output_shape = output_shape

    def forward(self, x):
        return self.mlp(self.backbone(x)).view(-1, *self.output_shape)

        

if __name__ == "__main__":
    model = Swin()
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)