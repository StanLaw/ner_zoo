from tqdm import tqdm
import random
import json
import torch

class BaseLoader:
	
	def __init__(self):
		self._data = []
	
	def load_from(self, filename):
		with open(filename, 'r') as f:
			for line in tqdm(f.readlines(), desc="data loading"):
				_datum = self._parse_line(line)
				self._data.append(_datum)

	def _parse_line(self, line):
		return json.loads(line)

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
		_names = []
		_label = []
		for one in input_list:
			_names.append(one.text)
			_label.append(one.label)
		return _names, _label

	def batch_iter(self, batch_size=64, seperate=False):
		self.shuffle()
		k = len(self._data) // batch_size
		for i in range(k):
			yield self._seperate(self._data[i * batch_size: (i+1) * batch_size])
		if len(self._data) % self.batch_size > 0:
			yield self._seperate(self._data[k * batch_size: ])

	
