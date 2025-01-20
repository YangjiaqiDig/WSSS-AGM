from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
import torch.nn.functional as F


class SegFormerModel(nn.Module):
    def __init__(self, n_classes=3, version="b2"):
        super(SegFormerModel, self).__init__()
        if version == "b5":
            self._size = 640
        else:
            self._size = 512
        model_name = (
            f"nvidia/segformer-{version}-finetuned-ade-{self._size}-{self._size}"
        )
        print("Loading model from", model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.decode_head.classifier = nn.Conv2d(
            768, n_classes, kernel_size=(1, 1), stride=(1, 1)
        )

        self.config = self.model.config

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)["logits"]
        outputs = F.interpolate(
            outputs, size=(self._size, self._size), mode="bilinear", align_corners=False
        )
        return outputs
