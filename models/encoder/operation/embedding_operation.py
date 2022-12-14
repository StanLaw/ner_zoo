from abc import ABCMeta, abstractmethod
from torch import nn, Tensor


class EmbeddingOperation(metaclass=ABCMeta):
    
	@abstractmethod
    def _embed_op(self, outputs, attention_mask=None):
        pass
    
	def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
