import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch import tensor

@dataclass
class BasicDataCollator:
	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
		return {
			"text_input": tensor([x["text_input"] for x in features]),
			"label_input": tensor([x["label_input"] for x in features]),
		}

@dataclass
class BaseDataCollator:
	def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
		text_input = self.tokenizer(
			[x["text"] for x in features],
			padding=True,
			return_tensor="pt",
			add_special_tokens=True,
			return_attention_mask=True,
			return_token_type_ids=False,
			truncation=True,
			max_length=self.max_length
		)
		batch_data = {
			"text_input": query_input, # 文本输入
			"label_input": None, # 标签输入
		}
		return batch_data
