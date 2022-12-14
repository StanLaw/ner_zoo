from .embedding_operation import EmbeddingOperation
from torch import nn
from transformers.activations import ACT2FN


class PoolingOperation(EmbeddingOperation):
    
	def __init__(self, config, model_args, output_embedding_size=192):
        self.model_args = model_args
        self.output_embedding_size = output_embedding_size
        if hasattr(config, "output_embedding_size"):
            self.output_embedding_size = config.output_embedding_size
        self.embeddingHead = nn.Linear(config.hidden_size, self.output_embedding_size)
        self.activation = ACT2FN['mish']
        #self.norm = nn.LayerNorm(self.output_embedding_size)
    
	def _embed_op(self, outputs, attention_mask=None):
        assert attention_mask is not None
        token_embeddings = self.activation(outputs.last_hidden_state)
        token_embeddings = self.embeddingHead(token_embeddings)
        return self.mean_pooling(token_embeddings, attention_mask)
    
	def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
	def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_over_time = torch.max(token_embeddings, 1)[0]
        return max_over_time
