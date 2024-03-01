import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Any


class MoEGate_Base(nn.Module):
	EPSILON = 1e-6
	def __init__(self, experts_shared, input_size, num_experts, num_nonzeros, prob_randperm = 0.9, flag_K_prob = True, training = True):
		super().__init__()
		self.experts_shared = experts_shared
		self.input_size = input_size
		self.num_experts = num_experts
		self.num_nonzeros = num_nonzeros
		self.prob_randperm = prob_randperm
		self.flag_K_prob = flag_K_prob
		self.training = training
		self.gating_nn = nn.Linear(self.input_size, self.num_experts, bias=False)

	def forward(self, x):
		gating_code = self._encode_gating_code(x)
		if self.training: gating_code = gating_code[:, torch.randperm(gating_code.shape[-1])] if np.random.random() > self.prob_randperm else gating_code
		if self.training: gating_code = gating_code + torch.rand(gating_code.shape[0], gating_code.shape[-1]).to(x.device)
		expert_weights, selector_outputs = self._compute_conditioned_expert_weights(gating_code)
		experts_shared_rep = torch.stack([e(x) for i, e in enumerate(self.experts_shared)])
		output = torch.einsum('ij, ji... -> i...', expert_weights, experts_shared_rep)
		if self.training: self._add_regularization_loss(expert_weights)
		return output, expert_weights
	
	#Output --> dim(expert_weights) = [N, num_experts], dim(selector_outputs) = [N, num_experts]
	def _compute_conditioned_expert_weights(self, gating_code):
		top_k_logits, top_k_indices = self._topK(gating_code, self.num_nonzeros, self.flag_K_prob if self.training else False)
		selector_outputs = torch.zeros_like(gating_code, device = gating_code.device).scatter(1, top_k_indices, 1.0)#.bool()
		expert_weights = F.softmax(top_k_logits, dim=1)
		expert_weights = torch.zeros_like(gating_code, device = gating_code.device).scatter(1, top_k_indices, expert_weights)
		return expert_weights, selector_outputs

	def _add_regularization_loss(self, expert_weights):
		loss = -(expert_weights*torch.log(expert_weights + self.EPSILON)).sum()
		loss.backward(retain_graph = True)
		pass

	def _encode_gating_code(self, x):
		return self.gating_nn(x)

	def _topK(self, gating_code, K, flag_K_prob):
		if flag_K_prob:
			index = torch.multinomial(F.softmax(gating_code, dim = -1), num_samples = K)
			values = torch.gather(gating_code, 1, index)
		else:
			values, index = gating_code.topk(k=K, dim=-1)
		return values, index
		
	
class TopKConvGate(MoEGate_Base):
	def __init__(self, **kwargs: Any):
		super().__init__(**kwargs)
		self.prob_randperm = 1.0
		self.gating_nn = nn.Sequential(
			nn.Conv2d(self.input_size, 128, kernel_size=3, stride=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Flatten(),
			nn.Linear(128, self.num_experts, bias=False)
		)
	

class TopKMLPGate(MoEGate_Base):
	def __init__(self, **kwargs: Any):
		super().__init__(**kwargs)
		self.prob_randperm = 1.0
		self.gating_nn = nn.Sequential(
			nn.Linear(self.input_size, 128, bias=False),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Linear(128, self.num_experts, bias=False),
		)


class TopKNonlinearMixGate(MoEGate_Base):
	def __init__(self, **kwargs: Any):
		super().__init__(**kwargs)
		self.gating_nn = nn.Conv2d(self.input_size, self.num_experts, kernel_size=3, stride=1)

	def forward(self, x):
		gating_code = self.gating_nn(x)
		gating_code = torch.sum(gating_code, 2)
		gating_code = torch.sum(gating_code, 2)
		if self.training:
			gating_code = gating_code + torch.rand(gating_code.shape[0], self.num_experts).to(x.device)
		expert_weights, dispatch_tensor = self._compute_conditioned_expert_weights(gating_code)
		expert_inputs = torch.einsum('bjkd,ben->ebjkd', x, dispatch_tensor)
		output = torch.stack([e(i) for e, i in zip(self.experts_shared, expert_inputs)])
		output = torch.einsum('ij, ji... -> i...', expert_weights, output)
		return output, expert_weights

	def _compute_conditioned_expert_weights(self, gating_code):
		gating_code = F.softmax(gating_code, dim=1)
		top_k_logits, top_k_indices = self._topK(gating_code, flag_m = "top1")
		mask = F.one_hot(top_k_indices, self.num_experts).float()
		mask_flat = mask.sum(dim=-1)
		combine_tensor = (top_k_logits[..., None, None] * mask_flat[..., None, None] * F.one_hot(top_k_indices, self.num_experts)[..., None])
		dispatch_tensor = combine_tensor.bool().to(combine_tensor)
		expert_weights = combine_tensor.squeeze(dim=-1)
		return expert_weights, dispatch_tensor

	def _topK(self, t, flag_m = "top1"):
		if flag_m == "top1":
			values, index = t.topk(k=1, dim=-1)
		elif flag_m == "top1_prob":
			index = t.multinomial(num_samples=1)
			values = torch.gather(t, 1, index)
		elif flag_m == "top2_prob":
			index = t.multinomial(num_samples=2)
			values = torch.gather(t, 1, index)
		values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
		return values, index