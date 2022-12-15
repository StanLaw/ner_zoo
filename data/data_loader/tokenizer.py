from transformers import BertTokenizer
from torch import tensor
import os
from .tokens import Str2TokenPath

class Tokenizer:
	def __init__(self, name="bert-base-chinese"):
		self._tok = BertTokenizer.from_pretrained(Str2TokenPath[name])

	def __call__(self, case, seq_length, to_tensor=False):

		if to_tensor:
			_tmp = self.__call__(case, seq_length, to_tensor=False)
			_res = {}
			for key, val in _tmp.items():
				_res[key] = tensor(val)
			return _res

		if isinstance(case, str):
			return self._tok(
						case, 
						max_length=seq_length, 
						padding=True,
						add_special_tokens=True,
						return_attention_mask=True,
						return_token_type_ids=False,
						truncation=True)
		if isinstance(case, list):
			_res = {}
			for one in case:
				tmp = self.__call__(one, seq_length)
				for key, val in tmp.items():
					if key not in _res:
						_res[key] = []
					_res[key].append(val)
			return _res
		raise Exception("input::case should be instance of str or list!")

