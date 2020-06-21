import numpy as np
import pickle
import logging

# PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My libraries
from config import load_config
from deeplearning import nn_registry
from deeplearning.dataset import UserDataset, assign_user_data
from grace_fl import compressor_registry
from grace_fl.gc_optimizer import grace_optimizer, LocalUpdater

def init_logger(config):
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

def test_accuracy(model, test_dataset, device="cuda"):
    
    server_dataset = UserDataset(test_dataset["images"], test_dataset["labels"])
    num_samples = test_dataset["labels"].shape[0]

    # Full Batch testing
    testing_data_loader = DataLoader(dataset=server_dataset, batch_size=len(server_dataset))
    for samples in testing_data_loader:
        results = model(samples["image"].to(device))
    
    predicted_labels = torch.argmax(results, dim=1).detach().cpu().numpy()
    accuracy = np.sum(predicted_labels == test_dataset["labels"]) / num_samples

    return accuracy

def train_accuracy(model, train_dataset, device="cuda"):

    server_dataset = UserDataset(train_dataset["images"], train_dataset["labels"])
    num_samples = train_dataset["labels"].shape[0]

    # Full Batch testing
    training_data_loader = DataLoader(dataset=server_dataset, batch_size=len(server_dataset))
    for samples in training_data_loader:
        results = model(samples["image"].to(device))
    
    predicted_labels = torch.argmax(results, dim=1).detach().cpu().numpy()
    accuracy = np.sum(predicted_labels == train_dataset["labels"]) / num_samples

    return accuracy

def train(config, logger, recoder):
    """Simulate Federated Learning training process. 
    
    Args:
        config (object class)
    """
    # initialize the model
    sampleSize = config.sampleSize[0] * config.sampleSize[1]
    classifier = nn_registry[config.model](dim_in=sampleSize, dim_out=config.classes)
    classifier.to(config.device)
    
    # Parse the configuration and fetch mode code for the optimizer
    mode = parse_config(config)

    # initialize data recoder 
    recoder["compress_ratio"] = []
    recoder["training_accuracy"] = []
    recoder["training_accuracy"] = []

    # initialize the optimizer for the server model
    optimizer = optim.SGD(params=classifier.parameters(), lr=config.lr)
    grace = compressor_registry[config.compressor]()
    optimizer = grace_optimizer(optimizer, grace, mode=mode) # wrap the optimizer
    
    dataset = assign_user_data(config)
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
                updater = LocalUpdater(userConfig)
                updater.local_step(classifier, optimizer, turn=iteration)
            
            optimizer.step()

            with torch.no_grad():
                # log train accuracy
                trainAcc = train_accuracy(classifier, dataset["trainData"], device=config.device)

                # validate the model and log test accuracy
                testAcc = test_accuracy(classifier, dataset["testData"], device=config.device)
                
                logger.info("Train accuarcy {:.8f}   Test accuracy {:.8f}".format(trainAcc, testAcc))

        recoder["compress_ratio"].append(optimizer.grace.compressRatio)
        logger.info("Averaged compression ratio: {:.8f}".format(recoder["compress_ratio"][-1]))
        optimizer.grace.reset()

def main():
    config = load_config()
    logger = init_logger(config)
    recoder = {}
    train(config, logger, recoder)
    
    with open("train_record.dat", "wb") as fp:
        pickle.dump(recoder, fp)

if __name__ == "__main__":
    main()


