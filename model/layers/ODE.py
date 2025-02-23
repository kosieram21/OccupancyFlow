import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint
from model.layers.SquashLinear import ConcatSquashLinear
	
class ODEFunc(nn.Module):
	def __init__(self, input_dim, condition_dim, hidden_dims, marginal):
		super(ODEFunc, self).__init__()

		self.marginal = marginal
		self.sampling_frequency = 1
		self.epsilon = None

		temporal_context_dim = 2 if marginal else 1
		dim_list = [input_dim] + list(hidden_dims) + [input_dim]
		
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(ConcatSquashLinear(dim_list[i], dim_list[i + 1], condition_dim + temporal_context_dim))
		self.layers = nn.ModuleList(layers)

	def _z_dot(self, t, z, condition):		
		if self.marginal:
			condition = condition.unsqueeze(1).expand(-1, z.shape[1], -1)
			time_encoding = t.expand(z.shape[0], z.shape[1], 1)
			positional_encoding = torch.cumsum(torch.ones_like(z)[:, :, 0], 1).unsqueeze(-1)
			positional_encoding = positional_encoding / self.sampling_frequency
			context = torch.cat([positional_encoding, time_encoding, condition], dim=-1)
		else:
			time_encoding = t.expand(z.shape[0], 1)
			context = torch.cat([time_encoding, condition], dim=-1)

		z_dot = z
		for l, layer in enumerate(self.layers):
			z_dot = layer(context, z_dot)
			if l < len(self.layers) - 1:
				z_dot = F.tanh(z_dot)
		return z_dot
	
	def forward(self, t, states):
		z = states[0]
		condition = states[2]

		with torch.set_grad_enabled(True):
			t.requires_grad_(True)
			for state in states:
				state.requires_grad_(True)
			z_dot = self._z_dot(t, z, condition)
			divergence = self._jacobian_trace(z_dot, z)

		return z_dot, -divergence, torch.zeros_like(condition).requires_grad_(True)
	
class ODE(nn.Module):
	def __init__(self):
		# implement the constructor
		self.t = 0
		
	def forward(self):
		# implement forward with ode_int
		return None