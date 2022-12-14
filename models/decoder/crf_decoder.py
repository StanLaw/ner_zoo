import torch.nn as nn
from torchcrf import CRF



class CrfDecoder(nn.Module):
	
	def __init__(self, labels):
		# labels : instance of data.labels
		self.model = CRF(len(labels), batch_first=True)
		self.labels = labels

	def forward(self, hidden, labels, mask=None):
		return self.model(hidden, labels, mask)

	def decode(self, hidden, mask):
		return self.model.viterbi_decode(hidden, mask)
		
