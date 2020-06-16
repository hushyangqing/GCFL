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

    return logger

def testAccuracy(model, testDataset, device="cuda"):
    
    serverDataset = userDataset(testDataset["images"], testDataset["labels"])
    numSamples = testDataset["labels"].shape[0]

    # Full Batch testing
    testingDataLoader = DataLoader(dataset=serverDataset, batch_size=serverDataset.__len__())
    for samples in testingDataLoader:
        results = model(samples.to("device"))
    
    predictedLabels = torch.argmax(results, dim=1).numpy()
    accuracy = (predictedLabels == testDataset["labels"]) / numSamples

    return accuracy

def trainAccuracy(model, trainDataset, device="cuda"):

    serverDataset = userDataset(trainDataset["images"], trainDataset["labels"])
    numSamples = trainDataset["labels"].shape[0]

    # Full Batch testing
    trainingDataLoader = DataLoader(dataset=serverDataset, batch_size=serverDataset.__len__())
    for samples in trainingDataLoader:
        results = model(samples.to("device"))
    
    predictedLabels = torch.argmax(results, dim=1).numpy()
    accuracy = (predictedLabels == trainDataset["labels"]) / numSamples

    return accuracy

def train(config, logger):
    # initialize the model
    sampleSize = config.sampleSize[0] * config.sampleSize[1]
    classifier = NNRegistry[config.model](dim_in=sampleSize, dim_out=config.classes)

    # initialize the optimizer
    optimizer = optim.SGD(params=classifier.parameters(), lr=config.lr)
    grace = compressorRegistry[config.compressor]()
    optimizer = graceOptimizer(optimizer, grace) # wrap the optimizer
    
    dataset = usersOwnData(config)
    iterationsPerEpoch = np.ceil(dataset["trainData"]["images"].shape[0]/config.localBatchSize)
    iterationsPerEpoch = iterationsPerEpoch.astype(np.int)

    if config.randomSampling:
        usersToSample = config.users * config.fa
        userIDs = np.arange(config.users) 

    for epoch in range(config.Epoch):
        logger.info("epoch {:.d}".format(epoch))
        for iteration in iterationsPerEpoch:
            # sample a fraction of users randomly
            if config.randomSampling:
                np.random.shuffle(userIDs)
                userIDs_candidates = userIDs[config.usersToSample]

            # Wait for all users aggregating gradients
            for userID in userIDs_candidates:
                sampleIDs = dataset["userWithData"][userID]
                userConfig = config.__dict__
                userConfig["images"] = dataset["trainData"]["images"][sampleIDs]
                userConfig["labels"] = dataset["trainData"]["labels"][sampleIDs]
                updater = localUpdater(userConfig)
                updater.localStep(classifier, optimizer)

                # log train accuracy
                trainAcc = trainAccuracy(classifier, dataset["testData"], device=config.device)
                logger.info("Train accuracy {:.4f}".format(trainAcc))

                # validate the model and log test accuracy
                testAcc = testAccuracy(classifier, dataset["testData"], device=config.device)
                logger.info("Test accuracy {:.4f}".format(testAcc))
    
            optimizer.step()


def main():
    config = loadConfig()
    logger = initLogger(config)
    train(config, logger)
    


if __name__ == "__main__":
    main()


