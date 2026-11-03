"""Supported torchvision model constructors."""

from torchvision import models


MODEL_REGISTRY = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "vit_b16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
    "convnext_t": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
}

