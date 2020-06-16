import logging

# PyTorch libraries
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Optimizer

# My libraries
from deeplearning import userDataset

class localUpdater(object):
    def __init__(self, **kwargs):
        self.criterion = nn.CrossEntropyLoss()
        
        try:
            self.lr = kwargs["lr"]
            self.batchSize = kwargs["batchSize"]
            self.localEpoch = kwargs["localEpoch"]
            self.device = kwargs["device"]

            assert("images" in kwargs)
            assert("labels" in kwargs)
        except [KeyError, AssertionError]:
            logging.error("LocalUpdater Initialization Failure! Input should include `lr`, `batchSize` and `samples`!") 

        self.sampleLoader = DataLoader(userDataset(kwargs["images"], kwargs["labels"]), 
                            batch_size=self.batchSize, 
                            shuffle=True
                            )
        self.dataVolumeInBit = 0

    def localStep(self, model, optimizer):

        # localEpoch is set to 1
        for i, sample in enumerate(self.sampleLoader):
            
            image = sample["image"].to(self.device)
            label = sample["label"].to(self.device)

            output = model(image)
            loss = self.criterion(output, label)
            loss.backward()



class _graceOptimzer(Optimizer):
    def __init__(self, params):
        pass



def graceOptimizer(optimizer, grace, named_parameters=None):
    """
    An optimizer that wraps another torch.optim.Optimizer.

    Allreduce operations are executed after each gradient is computed by ``loss.backward()``
    in parallel with each other. The ``step()`` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    Args:
        optimizer (torch.nn.optim.Optimizer):   Optimizer to use for computing gradients and applying updates.
        named_parameters (generator of dict):   A mapping between parameter names and values. 
        grace ():                               Compression algorithm used during allreduce to reduce the amount
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method.

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
        dict(_graceOptimzer.__dict__))
    return cls(optimizer.param_groups, named_parameters, grace)
