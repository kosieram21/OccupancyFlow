import torch
import torch.nn as nn

class FiLM(nn.Module):
	def __init__(self, dim_in, dim_out, dim_c):
		super(FiLM, self).__init__()
		self._layer = nn.Linear(dim_in, dim_out)
		self._hyper_bias = nn.Linear(dim_c, dim_out, bias=False)
		self._hyper_gate = nn.Linear(dim_c, dim_out)

	def forward(self, context, x):
		gate = torch.sigmoid(self._hyper_gate(context))
		bias = self._hyper_bias(context)
		ret = self._layer(x) * gate + bias
		return ret