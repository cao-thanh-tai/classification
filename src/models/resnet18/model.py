"""ResNet-18 classifier factory."""

from src.models.builder import build_model


def create_model(num_classes: int, pretrained: bool = True):
    return build_model("resnet18", num_classes, pretrained)
