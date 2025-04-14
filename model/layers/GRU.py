import torch
import torch.nn as nn

class GRU(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, bidirectional):
		super(GRU, self).__init__()
		self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

	def forward(self, x, mask=None):
		x = x * mask.unsqueeze(-1) if mask is not None else x
		embedding, _ = self.gru(x)
		return embedding[:, -1, :]