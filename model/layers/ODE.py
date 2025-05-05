import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from model.layers.FiLM import FiLM
	
class ODEFunc(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims, num_fourier_features):
		super(ODEFunc, self).__init__()
		self.num_fourier_features = num_fourier_features

		fourier_expanded_dim = input_dim + (input_dim * num_fourier_features)
		dim_list = [fourier_expanded_dim] + list(hidden_dims) + [input_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(FiLM(dim_list[i], dim_list[i + 1], condition_dim + 1))
		self.layers = nn.ModuleList(layers)

	def compute_positional_fourier_features(self, x):
		encodings = [x]
		for i in range(self.num_fourier_features // 2):
			freq = 2.0 ** i
			sin_features = torch.sin(freq * x)
			cos_features = torch.cos(freq * x)
			encodings.append(sin_features)
			encodings.append(cos_features)
		return torch.cat(encodings, dim=-1)

	def _h_dot(self, t, h, scene_context):
		if scene_context is not None:
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
	
	def forward(self, t, state):
		h = state[0]
		scene_context = state[1]
		h_fourier = self.compute_positional_fourier_features(h)
		h_dot = self._h_dot(t, h_fourier, scene_context)
		return h_dot, None
	
class ODE(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims, num_fourier_features):
		super(ODE, self).__init__()

		self.vector_field = ODEFunc(input_dim, condition_dim, hidden_dims, num_fourier_features)
		
	def forward(self, t, h, scene_context, mask=None):
		# TODO: how should we use the mask here?
		state = (h, scene_context)
		flow, _ = self.vector_field(t, state)
		return flow
	
	def solve_ivp(self, initial_value, scene_context, integration_times, mask=None):
		# TODO: implement warp occupancy as an initial value problem (IVP)
		state = (initial_value, scene_context)
		states = odeint_adjoint(self.time_derivative, state, integration_times, method='rk4', atol=1e-3, rtol=1e-3)
		return None