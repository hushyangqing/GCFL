import numpy as np
import pickle
import logging

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim

# My libraries
from config import loadConfig
from deeplearning import usersOwnData
from deeplearning.networks import naiveMLP

def initLogger(config):
    logLevel = config.logLevel
    logger = logging.getLogger(__name__)
    logger.setLevel(logLevel)

    fh = logging.FileHandler(config.logFile)
    fh.setLevel(logLevel)
    sh = logging.StreamHandler()
    sh.setLevel(logLevel)

def train(config):
    optimizer = optim.SGD(lr=config.lr)
    
    for epoch in range(config.Epoch):
        

def main():
    cfg = loadConfig()
    dataSet = usersOwnData(cfg)




if __name__ == "__main__":
    main()


