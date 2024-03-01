import torch.nn as nn
import torch
import torch.nn.functional as F
from MoE_layer import ResNetBlock_MoE
from torchvision.models.resnet import conv1x1


class ResNet18_MoEBlock(nn.Module):
    num_blocks = [2, 2, 2, 2]
    num_planes = [64, 128, 256, 512]
    num_planes_block = [[64,64],[64,64],[64,128],[128,128],[128,256],[256,256],[256,512],[512,512]]
    stride_block = [1,1,2,1,2,1,2,1]
    block = ResNetBlock_MoE
    def __init__(self, num_experts, top_k, n_classes=10):
        super(ResNet18_MoEBlock, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        #make the four layers
        self.layers = nn.ModuleList()
        for p, s in zip(self.num_planes_block, self.stride_block):
            downsample = None
            if s != 1 or p[0] != p[1]:
                downsample = nn.Sequential(conv1x1(p[0], p[1], s), nn.BatchNorm2d(p[1]))
            self.layers.append(self.block(in_planes = p[0], planes = p[1], stride = s, downsample = downsample, num_experts = self.num_experts, top_k = self.top_k))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_planes[3], n_classes)
        #init params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        [x, expert_weights_1] = self.layers[0](x)
        [x, expert_weights_2] = self.layers[1](x)
        [x, expert_weights_3] = self.layers[2](x)
        [x, expert_weights_4] = self.layers[3](x)
        [x, expert_weights_5] = self.layers[4](x)
        [x, expert_weights_6] = self.layers[5](x)
        [x, expert_weights_7] = self.layers[6](x)
        [x, expert_weights_8] = self.layers[7](x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        out = self.fc(feature)
        return out