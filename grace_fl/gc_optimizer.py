import logging

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Optimizer

# My libraries
from deeplearning import userDataset

class localUpdater(object):
    def __init__(self, userConfig):
        """Construct a local updater for a user.

        Args:
            lr (float):             learning rate for the user.
            batchSize (int):        batch size for the user. 
            localEpoch (int):       training epochs for the user.
            device (str):           set 'cuda' or 'cpu' for the user. 
            images (torch.Tensor):  training images of the user.
            labels (torch.Tensor):  training labels of the user.
        """
        
        
        try:
            self.lr = userConfig["lr"]
            self.batchSize = userConfig["localBatchSize"]
            self.device = userConfig["device"]

            assert("images" in userConfig)
            assert("labels" in userConfig)
        except [KeyError, AssertionError]:
            logging.error("LocalUpdater Initialization Failure! Input should include `lr`, `batchSize` and `samples`!") 

        self.sampleLoader = DataLoader(userDataset(userConfig["images"], userConfig["labels"]), 
                            batch_size=self.batchSize, 
                            shuffle=True
                            )
        self.criterion = nn.CrossEntropyLoss()

    def localStep(self, model, optimizer, **kwargs):

        # localEpoch is set to 1
        for i, sample in enumerate(self.sampleLoader):
            
            image = sample["image"].to(self.device)
            label = sample["label"].to(self.device)

            output = model(image)
            loss = self.criterion(output, label)
            loss.backward()
            optimizer.gather(**kwargs)


class _graceOptimizer(Optimizer):
    """
    An warpper optimizer gather gradients from local users and overwrite 
    step() method for the server.
    """
    def __init__(self, params, grace):
        super(self.__class__, self).__init__(params)
        self.rawBits = 0
        self.encodedBit = 0
        self.grace = grace
        self.grace.register(self.param_groups[0]["params"])

        self.gatheredGradients = []
        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                self.gatheredGradients.append(torch.zeros_like(param))

    def gather(self, **kwargs):
        """Gather local gradients and data volume in bits.
        """
        for group in self.param_groups:

            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                
                if self.grace._require_grad_idx == True:
                    kwargs["gradIdx"] = i
                    
                encodedTensor = self.grace.compress(param.grad.data, **kwargs)

                self.gatheredGradients[i] += self.grace.decompress(encodedTensor, shape=param.grad.data.shape)                
                
                # clear the gradients for next step, which is equivalent to zero_grad()
                param.grad.detach_()
                param.grad.zero_() 

    def setParamGroups(self, model):
        """
        Set the param_groups with the learnable paramters from a new model.

        Args:
            model (nn.torch.Module): the new model to be attached to the optimizer.
        """
        params = []
        for param in model.paramters():
            params.append(param)
        
        for group in self.param_groups:
            group["params"] = params

    def step(self, **kwargs):
        """Performs a single optimization step.
        """
        for group in self.param_groups:

            for i, param in enumerate(group['params']):

                d_param = self.grace.transAggregation(self.gatheredGradients[i], **kwargs)

                param.data.add_(-group['lr'], d_param)
 

def graceOptimizer(optimizer, grace):
    """
    An optimizer that wraps another torch.optim.Optimizer.

    Allreduce operations are executed after each gradient is computed by ``loss.backward()``
    in parallel with each other. The ``step()`` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    Args:
        optimizer (torch.nn.optim.Optimizer):   Optimizer to use for computing gradients and applying updates.
        grace (grace_fl.Compressor):            Compression algorithm used during allreduce to reduce the amount
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method.

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
        dict(_graceOptimizer.__dict__))
    return cls(optimizer.param_groups, grace)
