from transformers.trainer import Trainer
import logging


class NerTrainer(Trainer):

	def __init__(self, *args, **kwargs):
		logger.info("Initializing Ner Trainer")
		super(NerTrainer, self).__init__(*args, **kwargs)


	def comput_loss(self, model: nn.Module, 
						  inputs: Dict[str, Union[torch.Tensor, Any]],
						  return_outputs=False) -> torch.Tensor:
		loss = model(**inputs)
		return loss

	

