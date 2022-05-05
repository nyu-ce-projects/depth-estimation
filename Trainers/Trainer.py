

from Trainers.MaskTrainer import MaskTrainer 
from Trainers.BaseTrainer import BaseTrainer 
from Trainers.ESPCNTrainer import ESPCNTrainer
from Trainers.DPTrainer import DPTrainer

def getTrainer(config):
    if config['model_name']=='MASKDEPTH':
        return MaskTrainer(config)
    elif config['model_name']=='ESPCN':
        return ESPCNTrainer(config)
    elif config['model_name']=='DPT':
        return DPTrainer(config)
    else:
        return BaseTrainer(config)