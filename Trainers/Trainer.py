

from Trainers.MaskTrainer import MaskTrainer 
from Trainers.BaseTrainer import BaseTrainer 
from Trainers.ESPCNTrainer import ESPCNTrainer
from Trainers.IntrinsicsTrainer import IntrinsicsTrainer
from Trainers.MaskCamlessTrainer import MaskCamlessTrainer

def getTrainer(config):
    if config['model_name']=='MASKDEPTH':
        # nohup python -u train.py --model MASKDEPTH > output_mask.log &
        return MaskTrainer(config)
    elif config['model_name']=='ESPCN':
        # nohup python -u train.py --model ESPCN > output_espcn.log &
        return ESPCNTrainer(config)
    elif config['model_name']=='ESPCN12':
        # nohup python -u train.py --model ESPCN12 > output_espcn12.log &
        return ESPCNTrainer(config)
    elif config['model_name']=='CAMLESS':
        # nohup python -u train.py --model CAMLESS > output_camless.log &
        return IntrinsicsTrainer(config)
    elif config['model_name']=='MASKCAMLESS':
        # nohup python -u train.py --model MASKCAMLESS > output_maskcamless.log &
        return MaskCamlessTrainer(config)
    elif config['model_name']=='MASKCAMLESS_ESPCN' or config['model_name']=='MASKCAMLESS_ESPCN_WEATHER':
        # nohup python -u train.py --model MASKCAMLESS_ESPCN > output_maskcamless_espcn.log &
        return MaskCamlessTrainer(config)
    elif config['model_name']=='MONODEPTH2':
        # nohup python -u train.py --model MONODEPTH2 > output_monodepth2.log &
        return BaseTrainer(config)
    else:
        raise Exception("Model Configuration not available") 