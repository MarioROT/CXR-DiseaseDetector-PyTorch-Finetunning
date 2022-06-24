import torch
import torchvision.models as models
from torch import nn
from torchvision.models import efficientnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


def get_efficientnet_backbone(backbone_name: str):
    """
    Regresa una arquitectura base versión de EfficientNet pre-entrenada en ImageNet.
    Además remueve la capa de submuestreo promedio (average-pooling) y la capa
    lineal al final de la arquitectura.
    """
    if backbone_name == "efficientnet_b0":
        pretrained_model = models.efficientnet_b0(pretrained=True, progress=False)
        out_channels = 1280
    elif backbone_name == "efficientnet_b1":
        pretrained_model = models.efficientnet_b1(pretrained=True, progress=False)
        out_channels = 1280
    elif backbone_name == "efficientnet_b2":
        pretrained_model = models.efficientnet_b2(pretrained=True, progress=False)
        out_channels = 1408
    elif backbone_name == "efficientnet_b3":
        pretrained_model = models.efficientnet_b3(pretrained=True, progress=False)
        out_channels = 1536
    elif backbone_name == "efficientnet_b4":
        pretrained_model = models.efficientnet_b4(pretrained=True, progress=False)
        out_channels = 1792
    elif backbone_name == "efficientnet_b5":
        pretrained_model = models.efficientnet_b5(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == "efficientnet_b6":
        pretrained_model = models.efficientnet_b6(pretrained=True, progress=False)
        out_channels = 2304
    elif backbone_name == "efficientnet_b7":
        pretrained_model = models.efficientnet_b7(pretrained=True, progress=False)
        out_channels = 2560

    # print('Pretrained: ', pretrained_model) # Si se presenta problema con algun backbone revisar que se puedan quitar [:-2] linea abajo
    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
    # print('Backbone: ', backbone)
    backbone.out_channels = out_channels

    return backbone

def get_mobilenet_backbone(backbone_name: str):
    """
    Regresa una arquitectura base versión de MobileNet pre-entrenada en ImageNet.
    Además remueve la capa de submuestreo promedio (average-pooling) y la capa
    lineal al final de la arquitectura.
    """

    if backbone_name == "mobilenet_v2":
        pretrained_model = models.mobilenet_v2(pretrained=True, progress=False)
        out_channels = 1280
        backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
    elif backbone_name == "mobilenet_v3":
        pretrained_model = models.mobilenet_v3_large(pretrained=True, progress=False)
        out_channels = 1280
        backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])

    backbone.out_channels = out_channels

    return backbone

def get_resnet_backbone(backbone_name: str):
    """
    Regresa una arquitectura base versión de ResNet pre-entrenada en ImageNet.
    Además remueve la capa de submuestreo promedio (average-pooling) y la capa
    lineal al final de la arquitectura.
    """
    if backbone_name == "resnet18":
        pretrained_model = models.resnet18(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == "resnet34":
        pretrained_model = models.resnet34(pretrained=True, progress=False)
        out_channels = 512
    elif backbone_name == "resnet50":
        pretrained_model = models.resnet50(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == "resnet101":
        pretrained_model = models.resnet101(pretrained=True, progress=False)
        out_channels = 2048
    elif backbone_name == "resnet152":
        pretrained_model = models.resnet152(pretrained=True, progress=False)
        out_channels = 2048
    print('Pretrained Model: \n', pretrained_model)
    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
    backbone.out_channels = out_channels
    print('Backbone: \n', backbone)
    return backbone

def get_shufflenet_v2_backbone(backbone_name: str):
    """
    Regresa una arquitectura base versión de ShuffleNet V2 pre-entrenada en ImageNet.
    Además remueve la capa de submuestreo promedio (average-pooling) y la capa
    lineal al final de la arquitectura.
    """
    if backbone_name == "shufflenet_v2_x0_5":
        pretrained_model = models.shufflenet_v2_x0_5(pretrained=True, progress=False)
        out_channels = 1024
    elif backbone_name == "shufflenet_v2_x1_0":
        pretrained_model = models.shufflenet_v2_x1_0(pretrained=True, progress=False)
        out_channels = 1024
    elif backbone_name == "shufflenet_v2_x1_5":
        pretrained_model = models.shufflenet_v2_x1_5(pretrained=True, progress=False)
        out_channels = 1024
    elif backbone_name == "shufflenet_v2_x2_0":
        pretrained_model = models.shufflenet_v2_x2_0(pretrained=True, progress=False)
        out_channels = 2048

    # print('Pretrained: ', pretrained_model)
    backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
    # print('Backbone: ', backbone)
    backbone.out_channels = out_channels

    return backbone
