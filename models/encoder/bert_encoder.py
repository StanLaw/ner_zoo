from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from torch import nn


class BertEncoder(BertPreTrainedModel):
	
	def __init__(self, config):
		
		super(BertEncoder, self).__init__(config)
		self.bert = BertModel(config)
		self.classifier = nn.Linear(config.hidden_size, config.num_labels)

		self.init_weights()

	def forward(self, input_ids, attention_mask):
		outputs = self.bert(input_ids, attention_mask, return_dict=False)[0]
		logits = self.classifier(outputs)
		return logits
	
