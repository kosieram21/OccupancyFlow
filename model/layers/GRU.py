import torch.nn as nn

class GRU(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, bidirectional):
		super(GRU, self).__init__()
		self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

	def forward(self, x):
		embedding, _ = self.gru(x)
		return embedding[:, -1, :] # TODO: max pooling?