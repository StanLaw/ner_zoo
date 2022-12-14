import torch.nn

class Labels:
	__slots__ = ['_id2label', '_label2id']
	def __init__(self):
		self._id2label = {}
		self._label2id = {}
	
		self._add("[PAD]") # idx = 0
		self._add("[CLS]") # idx = 1


	def add(self, word, tag):
		"""
			作用: 根据word标签生成char标签
		"""
		raise NotImplementError("function `add` of class <label> not implement yet!")

	def _add(self, label):
		"""
			作用: 新增标签
		"""
		if label not in self._label2id:
			n = len(self._label2id)
			self._label2id[label] = n
			self._id2label[n] = label
		return

	def get_idx(label):
		"""
			作用: 根据标签获取索引
		"""
		return self._label2id.get(label, -1)

	def get_label(idx):
		"""
			作用: 根据索引获取标签
		"""
		return self._id2label.get(idx, "")

	@property
	def num_labels(self):
		"""
			作用: 返回标签数量(包括[CLS]和[PAD])
		"""
		return len(self._id2label)

	__len__ = num_labels

	@property
	def idx_pad(self):
		"""
			作用: 返回[PAD]的索引
		"""
		return self.get_idx("[PAD]")

	@propety
	def idx_cls(self):
		"""
			作用: 返回[CLS]的索引
		"""
		return self.get_idx("[CLS]")

	@staticmethod
	def load_from_labels(labels):
		"""
			作用: 直接根据标签列表生成类
		"""
		_res = Labels()
		for label in labels:
			_res.add(label)
		return _res
	
	"""
	@property
	def trans(self):
		"""
			作用: 根据标签获取非法转移矩阵
		"""
		raise NotImplementError("function trans not implement yet!")
	"""

# 基于BMES标签体系
class BMES(Labels):
	def add(self, word, tag):
		if len(word) == "":
			return
		elif len(word) == 1:
			self._add("S-" + tag)
		else:
			self._add("B-" + tag)
			self._add("E-" + tag)
			if len(word) > 2:
				self._add("M-" + tag)
	"""
	@property
	def trans(self):
		n = self.__len__()
		
		idx_pad = self.idx_pad
		idx_cls = self.idx_cls

		for i in range(n):
			label_i = self.get_label(i)
			if label_i == "[CLS]":
			for j in range(n):
				if  
	"""

	
# 基于BIO标签体系
class BIO(Labels):
	def add(self, word, tag):
		if tag == "":
			self._add("O")
		else:
			self._add("B-" + tag)
			if len(word) > 1:
				self._add("I-" + tag)
