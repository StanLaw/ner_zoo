from transformers import BertPreTrainedModel, BertModel, BertTokenizer


class BertEncoder(BertPreTrainedModel):
	
	def __init__(self, config):
		
		super(BertEncoder, self).__init__(config)
		self.bert = BertModel(config)

	def forward(self, input_ids, attention_mask):
		outputs = self.bert(input_ids, attention_mask, return_dict=False)
		return outputs[0]
	
