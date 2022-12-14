from .encoder import Str2Encoder
from .decoder import Str2Decoder
import torch.nn as nn


class BertCrfModel(nn.Module):

	def __init__(self, config):
		self.bert = Str2Encoder["BertEncoder"](config)
		self.bert.from_pretrained(config.init_path)
		self.crf = Str2Decoder["CrfDecoder"](config)

	def forward(self, features, labels):
		seq_output = self.bert(**features)
		loss = self.crf(seq_output, labels)
		return loss

	def predict(self, features):
		seq_output = self.bert(**features)
		labels = self.crf.decode(seq_output)
		return labels



