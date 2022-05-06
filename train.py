
import argparse

from Trainers.Trainer import getTrainer

from configs.config_loader import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mondep Configuration')
    parser.add_argument(
        "-c", "--conf", action="store", dest="conf_file",
        help="Path to config file"
    )
    parser.add_argument(
        "-m", "--model", action="store", dest="model",
        help="model name"
    )
    parser.add_argument("-tb", "--tensorboard", action="store_true", dest="tb_flag",help="tensorboard flag")
	parser.add_argument("-tbpth", "--tensorboard_path", action="store", dest="tb_path",help="tensorboard path")
	args = parser.parse_args()
    
    initiateTensorboard(args.tb_flag,args.tb_path)

    config = load_config(config_path=args.conf_file,model_name=args.model)
    net = getTrainer(config)

    net.train()
