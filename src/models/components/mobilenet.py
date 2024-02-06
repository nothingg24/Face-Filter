import torch
from torch import nn
import torchvision
from torchvision.models import list_models, get_model, get_weight

class MobileNet(nn.Module):
    def __init__(
            self,
            model_name: str = "mobilenet_v3_large",
            weights: str = "IMAGENET1K_V2",
            output_shape: list = [68, 2]):
        super().__init__()

        backbone = get_model(name=model_name, weights=None)
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, output_shape[0]*output_shape[1])
        
        self.backbone = backbone
        self.apply(self._init_weights)
        
        self.output_shape = output_shape
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)    

    def forward(self, x):
        return self.backbone(x).view(-1, *self.output_shape)

        

if __name__ == "__main__":
    model = MobileNet()
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)