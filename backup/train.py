from bert_crf_model import BertCrfModel
import sys, os
sys.path.append(os.path.dirname(__file__))
from data import BaseLoader, CmeeeDataPath
from argparser import DataArgument, ModelArgument
from transformers import HfArgumentParser


def main():
	parser = HfArgumentParser(DataArgument, ModelArgument)
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		data_args, model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		data_args, model_args = parser.parse_args_into_dataclasses()

	# remove useless handler in logging
	my_logger = logging.getLogger()
	for h in my_logger.handlers:
		my_logger.removeHandler(h)

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s-%%(levelname)s-%(name)s-%(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO
	)

	# init from ckpt
	init_model_path = None
	if (
		os.path.exists(model_args.output_dir)
		and any((x.startswith("checkpoint") for x in os.listdir(model_args.output_dir)))
	):
		if model_args.continue_train:
			ckpts = os.listdir(model_args.output_dir)
			ckpts = list(filter(lambda x: x.startswith("checkpoint"), ckpts))
			ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
			init_model_path = ckpts[-1]
		elif not model_args.overwrite_output_dir:
			raise ValueError(
				f"Output directory ({model_args.output_dir}) already exists and is not empty."
				"Use --overwrite_output_dir to overcome."
			)
	
	if not init_model_path:
		init_model_path = model_args.init_path
	
	logger.info(f"init model path: {init_model_path}")
	

	logger.info("DataArgument : %s", data_args)
	logger.info("ModelArgument: %s", model_args)

	# model load
	model = BertCrfModel(model_args)

	# data load
	train_data = BaseLoader(
		filename=CmeeeDataPath,
		seq_length=data_args.max_length
	)
	data_collator = BasicDataCollator()
	
	# trainer define 
	trainer = Trainer(
		model=model,
		args=model_args,
		train_dataset=train_data,
		data_collator=data_collator
	)

	trainer.add_callback(MyTrainerCallback(save_epoch_interval=1))
	trainer.train(resume_from_checkpoint=init_model_path)


if __name__ == "__main__":
	main()
