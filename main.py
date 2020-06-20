import numpy as np
import pickle
import logging

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My libraries
from config import loadConfig
from deeplearning import NNRegistry
from deeplearning.dataset import userDataset, usersOwnData
from deeplearning.networks import naiveMLP
from grace_fl import compressorRegistry
from grace_fl.gc_optimizer import graceOptimizer, localUpdater

def initLogger(config):
    """Initialize a logger object. 
    """
    logLevel = config.logLevel
    logger = logging.getLogger(__name__)
    logger.setLevel(logLevel)

    fh = logging.FileHandler(config.logFile)
    fh.setLevel(logLevel)
    sh = logging.StreamHandler()
    sh.setLevel(logLevel)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("-"*80)

    return logger

def parse_config(config):
    if config.predictive and config.takeTurns:
        mode = 0
    elif config.predictive:
        mode = 1
    elif config.takeTurns:
        mode = 2
    else:
        mode = 3

    return mode

def testAccuracy(model, testDataset, device="cuda"):
    
    serverDataset = userDataset(testDataset["images"], testDataset["labels"])
    numSamples = testDataset["labels"].shape[0]

    # Full Batch testing
    testingDataLoader = DataLoader(dataset=serverDataset, batch_size=serverDataset.__len__())
    for samples in testingDataLoader:
        results = model(samples["image"].to(device))
    
    predictedLabels = torch.argmax(results, dim=1).detach().cpu().numpy()
    accuracy = np.sum(predictedLabels == testDataset["labels"]) / numSamples

    return accuracy

def trainAccuracy(model, trainDataset, device="cuda"):

    serverDataset = userDataset(trainDataset["images"], trainDataset["labels"])
    numSamples = trainDataset["labels"].shape[0]

    # Full Batch testing
    trainingDataLoader = DataLoader(dataset=serverDataset, batch_size=serverDataset.__len__())
    for samples in trainingDataLoader:
        results = model(samples["image"].to(device))
    
    predictedLabels = torch.argmax(results, dim=1).detach().cpu().numpy()
    accuracy = np.sum(predictedLabels == trainDataset["labels"]) / numSamples

    return accuracy

def train(config, logger):
    # initialize the model
    sampleSize = config.sampleSize[0] * config.sampleSize[1]
    classifier = NNRegistry[config.model](dim_in=sampleSize, dim_out=config.classes)
    classifier.to(config.device)
    
    # Parse the configuration and fetch mode code for the optimizer
    mode = parse_config(config)

    # initialize the optimizer for the server model
    optimizer = optim.SGD(params=classifier.parameters(), lr=config.lr)
    grace = compressorRegistry[config.compressor]()
    optimizer = graceOptimizer(optimizer, grace, mode=mode) # wrap the optimizer
    
    dataset = usersOwnData(config)
    iterationsPerEpoch = np.ceil((dataset["trainData"]["images"].shape[0]*config.samplingFraction)/config.localBatchSize)
    iterationsPerEpoch = iterationsPerEpoch.astype(np.int)

    if config.randomSampling:
        usersToSample = int(config.users * config.samplingFraction)
        userIDs = np.arange(config.users) 
    
    for epoch in range(config.epoch):
        logger.info("epoch {:02d}".format(epoch))
        for iteration in range(iterationsPerEpoch):
            # sample a fraction of users randomly
            if config.randomSampling:
                np.random.shuffle(userIDs)
                userIDs_candidates = userIDs[:usersToSample]

            # Wait for all users aggregating gradients
            for userID in userIDs_candidates:
                sampleIDs = dataset["userWithData"][userID]
                userConfig = dict(config.__dict__)
                userConfig["images"] = dataset["trainData"]["images"][sampleIDs]
                userConfig["labels"] = dataset["trainData"]["labels"][sampleIDs]
                updater = localUpdater(userConfig)
                updater.localStep(classifier, optimizer, turn=iteration)
            
            optimizer.step()

            with torch.no_grad():
                # log train accuracy
                trainAcc = trainAccuracy(classifier, dataset["trainData"], device=config.device)

                # validate the model and log test accuracy
                testAcc = testAccuracy(classifier, dataset["testData"], device=config.device)
                
                logger.info("Train accuarcy {:.8f}   Test accuracy {:.8f}".format(trainAcc, testAcc))

        logger.info("Averaged compression ratio: {:.8f}".format(optimizer.grace.compressRatio))
        optimizer.grace.reset()
        
def main():
    config = loadConfig()
    logger = initLogger(config)
    train(config, logger)
    

if __name__ == "__main__":
    main()


