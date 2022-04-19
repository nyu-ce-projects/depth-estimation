
import argparse

from Trainer import Trainer

from utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mondep Configuration')
    parser.add_argument(
        "-c", "--conf", action="store", dest="conf_file",
        help="Path to config file"
    )
    args = parser.parse_args()
    conf = load_config(config_path=args.conf_file)
    net = Trainer()

    net.train()
