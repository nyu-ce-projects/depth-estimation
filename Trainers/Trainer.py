

from Trainers.MaskTrainer import MaskTrainer 
from Trainers.BaseTrainer import BaseTrainer 
from Trainers.ESPCNTrainer import ESPCNTrainer

def getTrainer(config):
    if config['model_name']=='MASKDEPTH':
        return MaskTrainer(config)
    elif config['model_name']=='ESPCN':
        return ESPCNTrainer(config)
    else:
        return BaseTrainer(config)