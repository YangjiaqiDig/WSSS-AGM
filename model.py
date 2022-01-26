import torch.nn as nn
import torchvision.models as models
import torch

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_class):
        super().__init__()
        self.base_model = backbone  # take the model without classifier
        # num_ftrs = self.base_model.fc.out_features
        in_ftrs = self.base_model.fc.in_features
        # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size>, <channels>, <width>, <height>]
        # so, let's do the spatial averaging: reduce <width> and <height> to 1
        # self.fc = nn.Linear(num_ftrs, num_class)    
        self.base_model.fc = nn.Linear(in_ftrs, num_class)
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        # f = torch.nn.Sequential(*list(self.base_model.children())[:-2])
        # print(f, f(x).shape)
        x = self.base_model(x)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
        # x = self.fc(x)
        return self.sigmoid(x)

if __name__ == "__main__":
    RES18 = models.resnet18(pretrained=True)
    model = MultiTaskModel(RES18, 5)
    # print(model)
    input_x = torch.rand(2, 3, 224, 224)
    output = model(input_x)
    print(output)