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
# print(model.__dict__['_modules'].keys())
# layers = list(model.children())[:-1]
# last_layer = list(model.children())[-1][2]
# print(model[-1][2].in_features)

class SimpleCNN(nn.Module):
    def __init__(
            self,
            model_name: str = "resnet18", # resnet50
            weights: str = "DEFAULT",
            output_shape: list = [68, 2]):
        super().__init__()

        backbone = get_model(name=model_name, weights=weights)
        backbone.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        
        # for param in backbone.parameters():
        #     param.requires_grad = False
        # for param in backbone.layer4.parameters():
        #     param.requires_grad = True
        supported = False
        if hasattr(backbone, 'fc'):
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, output_shape[0] * output_shape[1])
            # for p in backbone.fc.parameters():
            #     p.requires_grad = True
            supported = True
        elif hasattr(backbone, 'heads'):
            heads = getattr(backbone, 'heads')
            if hasattr(heads, 'head'):
                num_ftrs = backbone.heads.head.in_features
                backbone.heads.head = nn.Linear(num_ftrs, output_shape[0] * output_shape[1])
                # for p in backbone.heads.head.parameters():
                #     p.requires_grad = True
                supported = True

        if not supported:
            print("Model is not supported")
            exit(1)
        # layers = list(backbone.children())[:-1]
        # last_in = list(backbone.children())[-1][2].in_features
        # layers.append(nn.Linear(last_in, output_shape[0] * output_shape[1]))
        # backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.feature_extractor = nn.Sequential(*layers)
        self.backbone = backbone
        self.output_shape = output_shape

    def forward(self, x):
        # return self.backbone(x).reshape(x.size(0), self.output_shape[0], self.output_shape[1])
        return self.backbone(x).view(-1, *self.output_shape)

        

if __name__ == "__main__":
    # from torchinfo import summary
    model = SimpleCNN()
    # summary(model=model,
    #         input_size=(16, 3, 224, 224),
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)

        