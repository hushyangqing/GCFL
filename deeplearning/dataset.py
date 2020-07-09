import numpy as np
import pickle
import logging

# PyTorch Libraries
import torch
from torch.utils.data import Dataset

class UserDataset(Dataset):
    def __init__(self, images, labels):
        """Construct a user train_dataset and convert ndarray 
        """
        images = (images/255).astype(np.float32)
        labels = (labels).astype(np.int64)
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.num_samples = images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] 
        return dict(image=image, label=label)


def assign_data(train_dataset, iid=1, num_users=1, **kwargs):
    """
    Assign train_dataset to multiple users.

    Args:
        train_dataset (dict):     a train_dataset which contains training samples and labels. 
        iid (bool/int):     whether the train_dataset is allocated as iid or non-iid distribution.
        num_users (int):     the number of users.
        labels_per_user:      number of labels assigned to the user in no-iid setting.

    Returns:
        dict:  keys denote userID ranging from [0,...,num_users-1] and values are sampleID
               ranging from [0,...,num_samples]
    
    """
    
    try:
        num_samples = train_dataset["labels"].shape[0]
        samples_per_user = num_samples // num_users
    except KeyError:
        logging.error("Input train_dataset dictionary doesn't cotain key 'labels'.")

    user_with_data = {}
    userIDs = np.arange(num_users)
    sampleIDs = np.arange(num_samples)
    np.random.shuffle(userIDs)
    np.random.shuffle(sampleIDs)
    
    # Assign the train_dataset in an iid fashion
    if iid:
        for userID in userIDs:
            user_with_data[userID] = sampleIDs[userID*samples_per_user: (userID+1)*samples_per_user].tolist()

    # Assign the train_dataset in a no-iid fashion:
    # n = labels_per_user must be included in **kwargs (default:1) 
    # each user is assigned with num_samples/(n*num_users) samples
    else:
        idxs_ascending_labels = np.argsort(train_dataset["labels"])
        numLabels = train_dataset["labels"][idxs_ascending_labels[-1]] + 1   # num_of_labels = last_label + 1
        iter = 0
        current_label = 0
        num_of_samples_per_label = num_samples // numLabels
        for iter, userID in enumerate(userIDs):
            if iter*samples_per_user <= (current_label + 1)*num_of_samples_per_label:
                user_with_data[userID] = idxs_ascending_labels[userID*samples_per_user:(userID + 1)*samples_per_user].tolist()
            else:
                user_with_data[userID] = idxs_ascending_labels[userID*samples_per_user:(userID + 1)*num_of_samples_per_label].tolist()
                current_label += 1

    return user_with_data

def assign_user_data(config):
    """
    Load data and generate user_with_data dict given the configuration.

    Args:
        config (class):    a configuration class.
    
    Returns:
        dict: a dict contains train_data, test_data and user_with_data[userID:sampleID].
    """
    
    with open(config.train_data_dir, "rb") as fp:
        train_data = pickle.load(fp)
    
    with open(config.test_data_dir, "rb") as fp:
        test_data = pickle.load(fp)

    user_with_data = assign_data(train_dataset=train_data, 
                              iid=config.iid, 
                              num_users=config.users, 
                              labels_per_user=config.labels_per_user)

    return dict(train_data=train_data,
                test_data=test_data,
                user_with_data=user_with_data)

def assign_user_resource(config, userID, train_dataset, user_with_data):
    """Simulate one user resource by assigning one batch_size of data.
    """
    user_resource = {}
    batch_size = config.local_batch_size
    user_resource["lr"] = config.lr
    user_resource["device"] = config.device
    user_resource["batch_size"] = config.local_batch_size

    sampleIDs = user_with_data[userID][:batch_size]
    user_resource["images"] = train_dataset["images"][sampleIDs]
    user_resource["labels"] = train_dataset["labels"][sampleIDs]

    # As a batch of data has been fetched, it should be put at the end of the sampleIDs list
    user_with_data[userID][:batch_size], user_with_data[userID][-batch_size:] = \
    user_with_data[userID][-batch_size:], user_with_data[userID][:batch_size]

    return user_resource

