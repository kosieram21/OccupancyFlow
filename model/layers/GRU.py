import torch
import torch.nn as nn

class GRU(nn.Module):
	def __init__(self, input_dim, hidden_dim, bidirectional=True):
		super(GRU, self).__init__()
		self.hidden_dim = hidden_dim
		self.gru = nn.GRU(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=4,
			batch_first=True,
			bidirectional=bidirectional
		)

	def forward(self, x, mask=None):
		batch_size, num_agents, seq_len, input_dim = x.shape
		x_filled = torch.nan_to_num(x, nan=0.0)
		x_flat = x_filled.view(batch_size * num_agents, seq_len, input_dim)
		output, _ = self.gru(x_flat)
		embedding = output[:, -1, :].view(batch_size, num_agents, 2*self.hidden_dim)
		embedding = embedding * mask.unsqueeze(-1) if mask is not None else embedding
		return embedding