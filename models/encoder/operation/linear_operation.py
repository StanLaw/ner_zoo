from .embedding_operation import EmbeddingOperation
from torch import nn


class LinearOperation(EmbeddingOperation):

    def __init__(self, config, model_args):
        self.model_args = model_args
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        else:
            # 默认使用192维输出
            self.output_embedding_size = 192
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        #self.norm = nn.LayerNorm(self.output_embedding_size)
    
	def _embed_op(self, outputs, attention_mask=None):
        text_embeds = outputs.last_hidden_state[:, 0]
        #norm is important in IP
        #text_embeds = self.norm(self.embeddingHead(text_embeds))
        text_embeds = self.embeddingHead(text_embeds)
        return text_embeds
