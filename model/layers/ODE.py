import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint#_adjoint
from model.layers.FiLM import FiLM
	
class ODEFunc(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims, num_fourier_features, include_x=False):
		super(ODEFunc, self).__init__()
		self.num_fourier_features = num_fourier_features
		self.include_x = include_x # TODO: delete me

		if self.include_x:
			fourier_expanded_dim = input_dim + input_dim * num_fourier_features if num_fourier_features > 0 else input_dim
		else:
			fourier_expanded_dim = input_dim * num_fourier_features if num_fourier_features > 0 else input_dim
		
		dim_list = [fourier_expanded_dim] + list(hidden_dims) + [input_dim]
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
	
	def forward(self, t, state):
		h = state[0]
		scene_context = state[1]
		h_fourier = self.compute_positional_fourier_features(h) if self.num_fourier_features > 0 else h
		h_dot = self._h_dot(t, h_fourier, scene_context)
		return h_dot, torch.zeros_like(scene_context).requires_grad_(True) if scene_context is not None else None
	
	def forward2(self, t, h, scene_context):
		h_fourier = self.compute_positional_fourier_features(h) if self.num_fourier_features > 0 else h
		h_dot = self._h_dot(t, h_fourier, scene_context)
		return h_dot
	
class ODE(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims, num_fourier_features, include_x=False):
		super(ODE, self).__init__()

		self.vector_field = ODEFunc(input_dim, condition_dim, hidden_dims, num_fourier_features, include_x)
		
	def forward(self, t, h, scene_context, mask=None):
		# TODO: how should we use the mask here?
		state = (h, scene_context)
		flow, _ = self.vector_field(t, state)
		return flow
	
	def solve_ivp(self, initial_value, integration_times, scene_context, mask=None):
		state = (initial_value, scene_context)
		#states = odeint_adjoint(self.vector_field, state, integration_times, method='euler')
		states = odeint(self.vector_field, state, integration_times, method='euler')
		return states
	
	def solve_ivp2(self, initial_value, integration_times, scene_context, mask=None):
		state = (initial_value, scene_context)
		states = self.forward_euler(state, integration_times)
		return states
	
	def forward_euler(self, h0, integration_times):
		num_components = len(h0)
		hs = tuple([] for _ in range(num_components))
		h = h0
		
		for i in range(num_components):
			hs[i].append(h[i])
			
		for t0, t1 in zip(integration_times[:-1], integration_times[1:]):
			dt = t1 - t0
			dh = self.vector_field(t0, h)

			h = tuple(h_i + dt * dh_i for h_i, dh_i in zip(h, dh))
			
			for i in range(num_components):
				hs[i].append(h[i])
				
		hs = tuple(torch.stack(h_list, dim=0) for h_list in hs)
		return hs
	
	def solve_ivp3(self, initial_values, integration_times, scene_context, mask=None):
		states = self.forward_euler2(initial_values, integration_times, scene_context)
		return states
	
	def forward_euler2(self, initial_values, integration_times, scene_context):
		t0 = integration_times[0]
		h = torch.cat(initial_values[int(t0 * 10)], dim=0).unsqueeze(0)
		hs = [h]
			
		for t0, t1 in zip(integration_times[:-1], integration_times[1:]):
			dt = t1 - t0
			dh = self.vector_field.forward2(t0, h, scene_context)
			h = h + dt * dh

			if initial_values[int(t1 * 10)]:
				hx = torch.cat(initial_values[int(t1 * 10)], dim=0).unsqueeze(0)
				h = torch.cat([h, hx], dim=1)
			
			hs.append(h)

		return hs