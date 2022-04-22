import torch.nn as nn
import torchvision.models as models
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ClassificationModel(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        resnet = models.resnext50_32x4d(pretrained=True)
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features,
            out_features=num_class)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

if __name__ == "__main__":
    model = ClassificationModel(5)
    # print(model)
    input_x = torch.rand(2, 1, 224, 224)
    output = model(input_x)
    print(output, output.shape)

######## PREPARING THE IMAGES (FOR LATER)

# image_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
#     ])

# dataset = OCTDataset(["dataset_DR", "dataset_DME/1", "dataset_DME/3"], transform=image_transform)
# train_dataset, test_dataset = random_split(dataset, (1680, 426))

# train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1)
# val_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

# # print("The sample size of train dataset: ",len(train_dataset))
# # print("The sample size of test dataset: ",len(test_dataset))

# for x in train_loader:
#     break


# # print(x['image'][0].shape)
# # print(x['labels'][0])


# # everytime for image --> x[i]['image'] and for label --> x[i]['labels][0]
# # print("Size of an image: ", x["image"].shape)
# # print("Label of the first image: ", x['labels'][0])
# # print("Label tensor shape is: ", x['labels'][0].shape)

# ######## PLOT ONE IMAGE

# # plt.imshow(x["image"].view(1,224,224)[0], cmap='gray')
# # plt.savefig("gozde_temp_images/outputt.png")

# ######### CNN ARCHITECTURE

# class ClassificationModel(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         resnet = models.resnext50_32x4d(pretrained=True)
#         resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#         resnet.fc = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(in_features=resnet.fc.in_features,
#             out_features=n_classes)
#         )
#         self.base_model = resnet
#         self.sigm = nn.Sigmoid()

#     def forward(self, x):
#         return self.sigm(self.base_model(x))
    
# model_gmd = ClassificationModel()

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_gmd.parameters(), lr=0.005)