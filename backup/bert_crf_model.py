from transformers import BertModel, BertPreTrainedModel

import torch.nn as nn


class BertCrfModel(BertPreTrainedModel):

	def __init__(self, config):
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)
		self.crf = CRF(self.num_labels, batch_first=True)

	def _get_logits(self, features):
		seq_output = self.bert(**features["text_input"], return_dict=False)[0]
		logits = self.classifier(seq_outputs)
		return logits

	def forward(self, features):
		logits = self._get_logits(features)
		loss = self.crf(logits, features["label_input"])
		return loss

	def predict(self, features):
		logits = self._get_logits(features)
		labels = self.crf.decode(seq_output)
		return labels



