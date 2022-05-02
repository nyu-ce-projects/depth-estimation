import configparser
import argparse


def load_config(config_path,model_name):
    # loading .cfg config file from global scope
    if config_path is None:
        config_path = 'configs/model_config.cfg'
    configDict = {}
    config = configparser.ConfigParser()
    config.read(config_path)
    # for key in config:
    #     for nkey in config[key]:
    #         new_key = (key+"_"+nkey).upper()
    #         configDict[new_key] = config[key][nkey] 
    configDict =  {**config['GLOBAL'],**config[model_name.upper()]}
    configDict['model_name'] = model_name
    configDict['frame_ids'] = configDict['frame_ids'].split(",")
    print(configDict)
    # os.environ.update(configDict)
    return configDict