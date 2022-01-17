import torch.nn as nn
import torchvision.models as models
import torch

class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)  # take the model without classifier
        num_ftrs = self.base_model.fc.out_features
        # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size>, <channels>, <width>, <height>]
        # so, let's do the spatial averaging: reduce <width> and <height> to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # create separate classifiers for our outputs
        # self.edema = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=num_ftrs, out_features=2)
        # )
        self.dril = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )
        # self.ez = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=num_ftrs, out_features=2)
        # )
        self.rpe = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )
        self.hrd = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )
        self.rt = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )
        self.qDril = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )
        self.srf = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )
        self.irf = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )
        self.ezAtt = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )
        self.ezDis = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=num_ftrs, out_features=2)
        )

    def forward(self, x):
        x = self.base_model(x)
        for param in self.base_model.parameters():
            param.requires_grad = False
        return {
            # 'edema': self.edema(x),
            'dril': self.dril(x),
            # 'ez': self.ez(x),
            'rpe': self.rpe(x),
            'hrd': self.hrd(x), 
            'rt': self.rt(x), 
            'qDril': self.qDril(x), 
            'srf': self.srf(x), 
            'irf': self.irf(x), 
            'ezAtt': self.ezAtt(x), 
            'ezDis': self.ezDis(x)
        }

if __name__ == "__main__":
    model = MultiTaskModel()
    input_x = torch.rand(2, 3, 224, 224)
    output = model(input_x)
    print(output["ezAtt"].shape)