from transformers import BertTokenizer
import os
from .tokens import Str2TokenPath

class Tokenizer:
	def __init__(self, name="bert-base-chinese"):
		self._tok = BertTokenizer.from_pretrained(Str2TokenPath[name])

	def __

