from transformers import TraniningArguments
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional


@dataclass
class DataArgument:
	data_name: str = field()
	max_length: int = field()


@dataclass
class ModelArgument(TrainingArguments):
	model_name_or_path: Optional[str] = field()
	decoder_name: Optional[str] = field(default="CRF",
										metadata={"choices": ["CRF", "SoftMax"]})
	encoder_name: Optional[str] = field(default="Bert", 
										metadata={"choices": ["Bert", "BertBiLstm"]})


