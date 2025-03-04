import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from model.layers.SquashLinear import ConcatSquashLinear
	
class ODEFunc(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims):
		super(ODEFunc, self).__init__()

		dim_list = [input_dim] + list(hidden_dims) + [input_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(ConcatSquashLinear(dim_list[i], dim_list[i + 1], condition_dim + 1))
		self.layers = nn.ModuleList(layers)

	def _h_dot(self, t, h, scene_context):		
		time_context = t.expand(h.shape[0], 1)
		context = torch.cat([time_context, scene_context], dim=-1)

		h_dot = h
		for l, layer in enumerate(self.layers):
			h_dot = layer(context, h_dot)
			if l < len(self.layers) - 1:
				h_dot = F.tanh(h_dot)
		return h_dot
	
	def forward(self, t, states):
		h = states[0]
		scene_context = states[1]
		h_dot = self._h_dot(t, h, scene_context)
		return h_dot, torch.zeros_like(scene_context).requires_grad_(True)
	
class ODE(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims):
		super(ODE, self).__init__()

		self.vector_field = ODEFunc(input_dim, condition_dim, hidden_dims)
		
	def forward(self, t, h, scene_context):
		states = (h, scene_context)
		flow = self.vector_field(t, states)
		return flow
	
	def solve_ivp(self, initial_value, scene_context):
		# TODO: implement warp occupancy as an initial value problem (IVP)
		return None