import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.FiLM import FiLM

class ConditionedMLP(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims, output_dim, num_fourier_features, include_x=False):
		super(ConditionedMLP, self).__init__()
		self.num_fourier_features = num_fourier_features
		self.include_x = include_x # TODO: delete me

		if num_fourier_features > 0:
			fourier_expanded_dim = input_dim * (num_fourier_features + (1 if self.include_x else 0))
		else:
			fourier_expanded_dim = input_dim
		
		dim_list = [fourier_expanded_dim] + list(hidden_dims) + [output_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(FiLM(dim_list[i], dim_list[i + 1], condition_dim + 1))
		self.layers = nn.ModuleList(layers)

	def compute_positional_fourier_features(self, x):
		if self.include_x:
			encodings = [x]
		else:
			encodings = []
		for i in range(self.num_fourier_features // 2):
			freq = 2.0 ** i
			sin_features = torch.sin(freq * math.pi * x)
			cos_features = torch.cos(freq * math.pi * x)
			encodings.append(sin_features)
			encodings.append(cos_features)
		return torch.cat(encodings, dim=-1)

	def _h_dot(self, t, h, scene_context):
		if scene_context is not None:
			t = t.expand(h.shape[0], h.shape[1], 1) if len(t.shape) == 0 else t
			scene_context = scene_context.unsqueeze(1)
			scene_context = scene_context.expand(scene_context.shape[0], t.shape[1], scene_context.shape[2])
			context = torch.cat([t, scene_context], dim=-1)
		else:
			context = t

		h_dot = h
		for l, layer in enumerate(self.layers):
			h_dot = layer(context, h_dot)
			if l < len(self.layers) - 1:
				h_dot = F.tanh(h_dot)

		return h_dot
	
	def forward(self, t, h, scene_context):
		h_fourier = self.compute_positional_fourier_features(h) if self.num_fourier_features > 0 else h
		h_dot = self._h_dot(t, h_fourier, scene_context)
		h_dot = torch.sigmoid(h_dot) # TODO: should this be in a seperate occupancy estimate head class?
		return h_dot