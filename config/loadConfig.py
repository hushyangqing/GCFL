import json
import os

def loadConfig():
    """Load configurations"""
    currentPath = os.path.dirname(__file__)

    with open(os.path.join(currentPath, "config.json"), "r") as fp:
        configDict = json.load(fp)

    # Empty class for json loading
    class cfg: pass
    
    for key in configDict:
        setattr(cfg, key, configDict[key])

    return cfg