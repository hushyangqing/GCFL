import numpy as np
import pickle
import logging

# PyTorch Libraries
from torch.utils.data import Dataset

class userDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.numSamples = images.shape[0]

    def __len__(self):
        return self.numSamples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return dict(image=image, label=label)


def dataAssign(dataset, iid=1, numUsers=1, **kwargs):
    """
    Assign dataset to multiple users.

    Args:
        dataset (dict):     a dataset which contains training samples and labels. 
        iid (bool/int):     whether the dataset is allocated as iid or non-iid distribution.
        numUsers (int):     the number of users.
        labelsPerUser:      number of labels assigned to the user in no-iid setting.

    Returns:
        dict:  keys denote userID ranging from [0,...,numUsers-1] and values are sampleID
               ranging from [0,...,numSamples]
    
    """
    
    try:
        numSamples = dataset["labels"].shape[0]
        samplesPerUser = numSamples // numUsers
    except KeyError:
        logging.error("Input dataset dictionary doesn't cotain key 'labels'.")

    userWithData = {}
    userIDs = np.arange(numUsers)
    sampleIDs = np.arange(numSamples)
    np.random.shuffle(userIDs)
    np.random.shuffle(sampleIDs)
    
    # Assign the dataset in an iid fashion
    if iid:
        for userID in userIDs:
            userWithData[userID] = sampleIDs[userID*samplesPerUser: (userID+1)*samplesPerUser]

    # Assign the dataset in a no-iid fashion:
    # n = labelsPerUser must be included in **kwargs (default:1) 
    # each user is assigned with numSamples/(n*numUsers) samples
    else:
        IdxsForAscendingLabels = np.argsort(dataset["labels"])
        numLabels = dataset["labels"][IdxsForAscendingLabels[-1]] + 1   # num_of_labels = last_label + 1
        iter = 0
        currentLabel = 0
        numOfSamplesPerlabel = numSamples // numLabels
        for iter, userID in enumerate(userIDs):
            if iter*samplesPerUser <= (currentLabel+1)*numOfSamplesPerlabel:
                userWithData[userID] = IdxsForAscendingLabels[userID*samplesPerUser: (userID+1)*samplesPerUser]
            else:
                userWithData[userID] = IdxsForAscendingLabels[userID*samplesPerUser: (userID+1)*numOfSamplesPerlabel]
                currentLabel += 1

    return userWithData

def usersOwnData(config):
    """
    Load data and generate userWithData dict given the configuration.

    Args:
        config (class):    a configuration class.
    
    Returns:
        dict: a dict contains trainData, testData and userWithData.
    """
    
    with open(config.trainDataDir, "rb") as fp:
        trainData = pickle.load(fp)
    
    with open(config.testDataDir, "rb") as fp:
        testData = pickle.load(fp)

    userWithData = dataAssign(dataset = trainData, 
                              iid = config.iid, 
                              numUsers = config.users, 
                              labelsPerUser = config.labelsPerUser)

    return dict(trainData = trainData,
                testData = testData,
                userWithData = userWithData)
