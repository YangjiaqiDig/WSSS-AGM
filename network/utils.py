import torchvision.models


def network_class(args):
    if args.backbone == "resnet18":
        print("Backbone: ResNet18")
        backbone = torchvision.models.resnet18(pretrained=True)
    elif args.backbone == "vgg16":
        print("Backbone: VGG16")
        backbone = torchvision.models.vgg16(pretrained=True)
    elif args.backbone == "resnet50":
        print("Backbone: ResNet50")
        backbone = torchvision.models.resnet50(pretrained=True)
    elif args.backbone == "resnet101":
        print("Backbone: ResNet101")
        backbone = torchvision.models.resnet101(pretrained=True)
    elif args.backbone == "densenet":
        print("Backbone: DenseNet")
        backbone = torchvision.models.densenet121(pretrained=True)
    elif args.backbone == "mnasnet":
        print("Backbone: MnasNet")
        backbone = torchvision.models.mnasnet1_0(pretrained=True)
    else:
        raise NotImplementedError("No backbone found for '{}'".format(args.backbone))
    return backbone


num_channels_fc = {
    "resnet18": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "vgg16": 512,
    "densenet": 1024,
    'mnasnet': 1280,
}
