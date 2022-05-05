

from Trainers.MaskTrainer import MaskTrainer 
from Trainers.BaseTrainer import BaseTrainer 
from Trainers.ESPCNTrainer import ESPCNTrainer
from Trainers.IntrinsicsTrainer import IntrinsicsTrainer

def getTrainer(config):
    if config['model_name']=='MASKDEPTH':
        # nohup python -u train.py --model MASKDEPTH > output_mask.log &
        return MaskTrainer(config)
    elif config['model_name']=='ESPCN':
        # nohup python -u train.py --model ESPCN > output_espcn.log &
        return ESPCNTrainer(config)
    elif config['model_name']=='CAMLESS':
        # nohup python -u train.py --model CAMLESS > output_camnet.log &
        return IntrinsicsTrainer(config)
    elif config['model_name']=='MONODEPTH2':
        # nohup python -u train.py --model MONODEPTH2 > output_monodepth2.log &
        return BaseTrainer(config)
    else:
        raise Exception("Model Configuration not available") 