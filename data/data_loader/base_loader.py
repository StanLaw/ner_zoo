from tqdm import tqdm
import random
import json
import os
from transformers import BertTokenizer
from .tokenizer import Tokenizer
from torch import tensor
from torch.utils.data import Dataset, Iterable

class BaseLoader(Dataset):
	
	def __init__(self, filename=None, seq_length=32, token_type="bert-base-chinese"):
		super(BaseLoader, self).__init__()
		self._data = []
		self.seq_length = seq_length
		self.label2id = {}
		self.tokenizer = Tokenizer(token_type)
		# 加载数据
		self.load_from(filename)
	
	def load_from(self, filename):
		# 加载标签
		with open(os.path.join(os.path.dirname(filename), "labels.txt"), 'r') as f:
			for line in f.readlines():
				line = line.strip()
				if line == "":
					continue
				self.label2id[line] = len(self.label2id)
		# 加载数据
		with open(filename, 'r') as f:
			for line in tqdm(f.readlines(), desc="data loading"):
				_datum = self._parse_line(line)
				self._data.append(_datum)

	def __parse_labels(self, labels):
		k = len(labels)
		_res = [1]
		for i in range(min(k, self.seq_length-2)):
			_res.append(self.label2id[labels[i]])
		k = len(_res)
		for i in range(k, self.seq_length):
			_res.append(0)
		return _res

	def _parse_line(self, line):
		datum = json.loads(line)
		_res = {
			"text_input": self.tokenizer(datum["text"], self.seq_length, to_tensor=False),
			"label_input": self.__parse_labels(datum["labels"])
		}
		return _res

	def __getitem__(self, k):
		return self._data[k]

	def train_test_split(self, test_rate=0.1):
		test_cnt = int(self.size * test_rate)
		self.shuffle()
		test_data, train_data = BaseLoader(), BaseLoader()
		test_data._data, train_data._data = self._data[: test_cnt], self._data[test_cnt: ]
		return train_data, test_data

	@property
	def size(self):
		return len(self._data)

	def shuffle(self):
		random.shuffle(self._data)

	@staticmethod
	def _seperate(input_list, flag):
		if not flag:
			return input_list

		_res = {}
		for one in input_list:
			for key in ["label_ids", "input_ids", "attention_mask"]:
				if key not in _res:
					_res[key] = []
				_res[key].append(one[key])
		for key in ["label_ids", "input_ids", "attention_mask"]:
			_res[key] = tensor(_res[key])
		
		return _res

	def batch_iter(self, batch_size=64, seperate=True):
		self.shuffle()
		k = len(self._data) // batch_size
		for i in range(k):
			yield self._seperate(self._data[i * batch_size: (i+1) * batch_size], seperate)
		if len(self._data) % batch_size > 0:
			yield self._seperate(self._data[k * batch_size: ], seperate)

	
