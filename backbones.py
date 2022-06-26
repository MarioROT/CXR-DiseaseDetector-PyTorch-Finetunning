import torch
import torchvision.models as models
from torch import nn
from torchvision.models import efficientnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# Use the torchvision's implementation of ResNeXt, but add FC layer for a different number of classes (27) and a Sigmoid instead of a default Softmax.
class GetModel(nn.Module):
    def __init__(self, n_classes, backbone_net):
        super().__init__()
        resnet = backbones.get_backbone(backbone_net)
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))


def get_backbone(backbone_name: str):
    """
    Regresa una arquitectura base versión de EfficientNet, ShuffleNet v2,
    MobileNet v2 o v3, EfficientNet pre-entrenada en ImageNet.
    """

    if backbone_name == "mobilenet_v2":
        pretrained_model = models.mobilenet_v2(pretrained=True, progress=False)
        out_channels = 1280
    elif backbone_name == "mobilenet_v3":
        pretrained_model = models.mobilenet_v3_large(pretrained=True, progress=False)
        out_channels = 1280
    elif backbone_name == "resnet18":
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
    elif backbone_name == "resnext50_32x4d":
        pretrained_model = models.resnext50_32x4d(pretrained=True)
    elif backbone_name == "shufflenet_v2_x0_5":
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
    # elif backbone_name == "efficientnet_b0":
    #     pretrained_model = models.efficientnet_b0(pretrained=True, progress=False)
    #     out_channels = 1280
    # elif backbone_name == "efficientnet_b1":
    #     pretrained_model = models.efficientnet_b1(pretrained=True, progress=False)
    #     out_channels = 1280
    # elif backbone_name == "efficientnet_b2":
    #     pretrained_model = models.efficientnet_b2(pretrained=True, progress=False)
    #     out_channels = 1408
    # elif backbone_name == "efficientnet_b3":
    #     pretrained_model = models.efficientnet_b3(pretrained=True, progress=False)
    #     out_channels = 1536
    # elif backbone_name == "efficientnet_b4":
    #     pretrained_model = models.efficientnet_b4(pretrained=True, progress=False)
    #     out_channels = 1792
    # elif backbone_name == "efficientnet_b5":
    #     pretrained_model = models.efficientnet_b5(pretrained=True, progress=False)
    #     out_channels = 2048
    # elif backbone_name == "efficientnet_b6":
    #     pretrained_model = models.efficientnet_b6(pretrained=True, progress=False)
    #     out_channels = 2304
    # elif backbone_name == "efficientnet_b7":
    #     pretrained_model = models.efficientnet_b7(pretrained=True, progress=False)
    #     out_channels = 2560

    return pretrained_model #backbone
