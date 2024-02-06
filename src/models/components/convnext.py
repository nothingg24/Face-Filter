import torch
from torch import nn
import torchvision
from torchvision.models import list_models, get_model, get_weight

# available_models = list_models()
# print(available_models)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
# model = torchvision.models.convnext_tiny(weights=weights)
# model = get_model(name="convnext_tiny", weights="DEFAULT")
# print(model)
# print(model.__dict__['_modules'].keys())
# layers = list(model.children())[:-1]
# last_layer = list(model.children())[-1][2]
# print(model[-1][2].in_features)

class SimpleConvnext(nn.Module):
    def __init__(
            self,
            model_name: str = "convnext_small", # tiny/small
            weights: str = "DEFAULT",
            output_shape: list = [68, 2]):
        super().__init__()

        backbone = get_model(name=model_name, weights=None)
        layers = list(backbone.children())[:-2]
        self.features = nn.Sequential(*layers)
        # for param in self.features.parameters():
        #     param.requires_grad = False
        # self.pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, output_shape[0] * output_shape[1])
        )
        
        # for param in backbone.parameters():
        #     param.requires_grad = False
        # backbone.classifier[2]= nn.Linear(backbone.classifier[2].in_features, output_shape[0] * output_shape[1])
        
        self.output_shape = output_shape

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x).view(-1, *self.output_shape)

        

if __name__ == "__main__":
    model = SimpleConvnext()
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)