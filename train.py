from models import BertCrfModel
from data import BaseLoader, CmeeeDataPath


def main(config):

	init_path = config.init_path
	save_path = config.save_path
	
	model = BertCrfModel(config)
	data = BaseLoader(config.bert_name)
	data.load_from(CmeeeDataPath)

	trainer = 

