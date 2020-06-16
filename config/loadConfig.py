import yaml
import json
import os

def loadJsonConfig():
    """Load configurations of json file"""
    currentPath = os.path.dirname(__file__)

    with open(os.path.join(currentPath, "config.json"), "r") as fp:
        configDict = json.load(fp)

    # Empty class for json loading
    class cfg: pass
    
    for key in configDict:
        setattr(cfg, key, configDict[key])

    return cfg

def loadConfig():
    """Load configurations of yaml file"""
    currentPath = os.path.dirname(__file__)

    with open(os.path.join(currentPath, "config.yaml"), "r") as fp:
        configDict = yaml.load(fp)

    # Empty class for yaml loading
    class cfg: pass
    
    for key in configDict:
        setattr(cfg, key, configDict[key])

    return cfg