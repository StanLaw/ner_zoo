from transformers import BertTokenizer
import os
from .tokens import Str2TokenPath

class Tokenizer:
	def __init__(self, name="bert-base-chinese"):
		self._tok = BertTokenizer.from_pretrained(Str2TokenPath[name])

	def __call__(self, case, seq_length):
		if isinstance(case, str):
			return self._tok.encode_plus(case, max_length=seq_length, padding="max_length")
		if isinstance(case, list):
			_res = {}
			for one in case:
				tmp = self.__call__(one, seq_length)
				for key, val in tmp:
					if key not in _res:
						_res[key] = []
					_res[key].append(val)
			return _res
		raise Exception("input::case should be instance of str or list!")

