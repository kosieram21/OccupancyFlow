import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from model.layers.Spline import NaturalCubicSpline

class CDEFunc(nn.Module):
	def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=3): # TODO: use num_layers
		super(CDEFunc, self).__init__()
		self.input_dim = input_dim
		self.embedding_dim = embedding_dim
		
		dim_list = [embedding_dim] + [hidden_dim] * num_layers + [input_dim * embedding_dim]
		layers = []
		for i in range(len(dim_list) - 1):
			layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
			if i < len(dim_list) - 2:
				layers.append(nn.ReLU())
		self.mlp = nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.mlp(x)
		x = x.tanh()
		x = x.view(*x.shape[:-1], self.embedding_dim, self.input_dim)
		return x
	

class VectorField(torch.nn.Module):
	def __init__(self, dX_dt, f):
		super(VectorField, self).__init__()
		self.dX_dt = dX_dt
		self.f = f

	def forward(self, t, z):
		dX_dt = self.dX_dt(t)
		f = self.f(z)
		out = (f @ dX_dt.unsqueeze(-1)).squeeze(-1)
		return out
	

class CDE(nn.Module):
	def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers):
		super(CDE, self).__init__()
		self.embed = torch.nn.Linear(input_dim, embedding_dim)
		self.f = CDEFunc(input_dim, embedding_dim, hidden_dim, num_layers)

	def forward(self, t, x, mask=None):
		if mask is not None:
			batch_size, max_agents, _, _ = x.shape
			embedding = torch.zeros(batch_size, max_agents, self.embed.out_features).to(x.device)
			valid = [x[i, j] for i in range(batch_size) for j in range(max_agents) if mask[i, j]]
			indicies = [(i, j) for i in range(batch_size) for j in range(max_agents) if mask[i, j]]
			x = torch.stack(valid, dim=0)

		spline = NaturalCubicSpline(t, x)
		vector_field = VectorField(dX_dt=spline.derivative, f=self.f)
		z0 = self.embed(spline.evaluate(t[0]))
		out = odeint_adjoint(vector_field, z0, t, method='rk4', atol=1e-3, rtol=1e-3)

		if mask is not None:
			embedding[[i for i, j in indicies], [j for i, j in indicies]] = out[-1]
		else:
			embedding = out[-1]

		return embedding