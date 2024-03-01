import torch.nn as nn
import numpy as np
from MoE_gate import DSelectKGate, TopKConvGate, TopKMLPGate, TopKNonlinearMixGate
from torchvision.models.resnet import BasicBlock


class BaseMoELayer(nn.Module):
    def __init__(self, num_experts, top_k, gate, experts):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = experts
        self.gate = gate

    def forward(self, x):
        if self.num_experts == 1:
            out = self.experts[0](x)
            return [out, None]
        #self.experts = experts try dropout
        out, expert_weights = self.gate(x)
        return [out, expert_weights]


class Conv_MoE(BaseMoELayer):
    def __init__(self, in_planes, out_planes, stride, num_experts, top_k):
        experts = nn.ModuleList([nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=3, padding=1, groups=1, bias=False, dilation=1) for i in range(num_experts)])
        gate = TopKConvGate(experts_shared = experts, input_size = in_planes, num_experts = num_experts, num_nonzeros = top_k) if num_experts > 1 else None
        super().__init__(num_experts=num_experts, top_k=top_k, gate=gate, experts=experts)


class Linear_MoE(BaseMoELayer):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        experts = nn.ModuleList([nn.Linear(in_dim, out_dim) for i in range(num_experts)])
        gate = TopKMLPGate(experts_shared = experts, input_size = in_dim, num_experts = num_experts, num_nonzeros = top_k) if num_experts > 1 else None
        super().__init__(num_experts=num_experts, top_k=top_k, gate=gate, experts=experts)


class ResNetBlock_MoE(BaseMoELayer):
    def __init__(self, in_planes, planes, stride, downsample, num_experts, top_k):
        experts = nn.ModuleList([BasicBlock(in_planes, planes, stride, downsample) for i in range(num_experts)])
        # gate = DSelectKGate(experts_shared = experts, input_size = 512, num_experts = num_experts, num_nonzeros = top_k)
        gate = TopKConvGate(experts_shared = experts, input_size = in_planes, num_experts = num_experts, num_nonzeros = top_k) if num_experts > 1 else None
        # gate = TopKMLPGate(experts_shared = experts, input_size = 512, num_experts = num_experts, num_nonzeros = top_k)
        # gate = TopKNonlinearMixGate(experts_shared = experts, input_size = in_planes, num_experts = num_experts, num_nonzeros = top_k)
        super().__init__(num_experts=num_experts, top_k=top_k, gate=gate, experts=experts)

        

        