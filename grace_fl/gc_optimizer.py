import logging

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import Optimizer

# My libraries
import grace_fl.constant as const
from deeplearning import UserDataset

class LocalUpdater(object):
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
            self.buffer = userConfig["device"]

            assert("images" in userConfig)
            assert("labels" in userConfig)
        except [KeyError, AssertionError]:
            logging.error("LocalUpdater Initialization Failure! Input should include `lr`, `batchSize` and `samples`!") 

        self.sampleLoader = DataLoader(UserDataset(userConfig["images"], userConfig["labels"]), 
                            batch_size=self.batchSize, 
                            shuffle=True
                            )
        self.criterion = nn.CrossEntropyLoss()

    def local_step(self, model, optimizer, **kwargs):

        # localEpoch and iteration is set to 1
        for sample in self.sampleLoader:
            
            image = sample["image"].to(self.device)
            label = sample["label"].to(self.device)

            output = model(image)
            loss = self.criterion(output, label)
            loss.backward()
            optimizer.gather(**kwargs)

            break

class _graceOptimizer(Optimizer):
    """
    A warpper optimizer gather gradients from local users and overwrite 
    step() method for the server.

    Args:
        params (nn.Module.parameters): model learnable parameters.
    """
    def __init__(self, params, grace, **kwargs):
        super(self.__class__, self).__init__(params)
        self.rawBits = 0
        self.encodedBit = 0
        self.grace = grace

        self._gatheredGradients = []
        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                self._gatheredGradients.append(torch.zeros_like(param))

    def gather(self, **kwargs):
        """Gather local gradients.
        """
        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                    
                encodedTensor = self.grace.compress(param.grad.data, **kwargs)

                self._gatheredGradients[i] += self.grace.decompress(encodedTensor, shape=param.grad.data.shape)                
                
                # clear the gradients for next step, which is equivalent to zero_grad()
                param.grad.detach_()
                param.grad.zero_() 

    def step(self, **kwargs):
        """Performs a single optimization step.
        """
        for group in self.param_groups:

            for i, param in enumerate(group['params']):

                d_param = self.grace.transAggregation(self._gatheredGradients[i], **kwargs)
                param.data.add_(-group['lr'], d_param)
                self._gatheredGradients[i].zero_()
    

class _predTurnOptimizer(Optimizer):
    """
    A warpper optimizer which implements predictive encoding with turn trick.
    It gather gradients residuals ("+" residual for even turns and "-" residual for 
    odd turns) from local users and overwrite step() method for the server.

    Args:
        params (nn.Module.parameters): model learnable parameters.
    """
    def __init__(self, params, grace, **kwargs):
        super(self.__class__, self).__init__(params)
        self.rawBits = 0
        self.encodedBit = 0
        self.grace = grace

        self.grace._current_sign = 1
        self._gatheredGradients = []
        self._plus_sign_buffer = []
        self._minus_sign_buffer = []
        self._buffer_empty = True
        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                self._gatheredGradients.append(torch.zeros_like(param))
                self._plus_sign_buffer.append(torch.zeros_like(param))
                self._minus_sign_buffer.append(torch.zeros_like(param))

    def gather(self, **kwargs):
        """Gather local gradients.
        """
        try:
            self.turn = kwargs["turn"]
            self.grace._current_sign = 1 if self.turn%2 == 0 else -1
        except KeyError:
            logging.error("Turn trick cannot be applied without 'turn' parameters.")

        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                

                # if buffer is empty, encode the gradient
                if self._buffer_empty:
                    encodedTensor = self.grace.compress(param.grad.data, turn=self.turn)
                    self._gatheredGradients[i] += self.grace.decompress(encodedTensor, shape=param.grad.data.shape)
                # if buffer in nonempty, encode the residual
                elif self.grace._current_sign == 1:
                    encodedTensor = self.grace.compress_with_reference(param.grad, self._plus_sign_buffer[i])
                    self._gatheredGradients[i] += self.grace.decompress_with_reference(encodedTensor, self._plus_sign_buffer[i])
                else:
                    encodedTensor = self.grace.compress_with_reference(param.grad, self._minus_sign_buffer[i])
                    self._gatheredGradients[i] += self.grace.decompress_with_reference(encodedTensor, self._minus_sign_buffer[i])

                if self.grace._current_sign == 1:
                    self._minus_sign_buffer[i] += (param.grad.data < -const.EPSILON)
                else:
                    self._plus_sign_buffer[i] += (param.grad.data > const.EPSILON)

                # clear the gradients for next step, which is equivalent to zero_grad()
                param.grad.detach_()
                param.grad.zero_() 


    def step(self):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                d_param = self.grace.transAggregation(self._gatheredGradients[i])

                # register buffer
                if self.grace._current_sign == 1:
                    self._plus_sign_buffer[i].zero_()
                    self._minus_sign_buffer[i] = self.grace.transAggregation(self._minus_sign_buffer[i])            
                else:
                    self._minus_sign_buffer[i].zero_()
                    self._plus_sign_buffer[i] = self.grace.transAggregation(self._plus_sign_buffer[i])
                
                param.data.add_(-group["lr"], d_param)
                self._gatheredGradients[i].zero_()
                
        self._buffer_empty = False

class _predSkipStepOptimizer(Optimizer):
    """
    A warpper optimizer which implements predictive encoding with turn trick.
    It gather gradients residuals ("+" residual for even turns and "-" residual for 
    odd turns) from local users and overwrite step() method for the server.

    Args:
        params (nn.Module.parameters): model learnable parameters.
    """
    def __init__(self, params, grace, **kwargs):
        super(self.__class__, self).__init__(params)
        self.rawBits = 0
        self.encodedBit = 0
        self.grace = grace

        self.grace._current_sign = 1
        self._gatheredGradients = []
        self._plus_sign_buffer = []
        self._minus_sign_buffer = []
        self._buffer_empty = True
        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                self._gatheredGradients.append(torch.zeros_like(param))
                self._plus_sign_buffer.append(torch.zeros_like(param))
                self._minus_sign_buffer.append(torch.zeros_like(param))

    def gather(self, **kwargs):
        """Gather local gradients.
        """
        try:
            self.turn = kwargs["turn"]
            self.grace._current_sign = 1 if self.turn%2 == 0 else -1
        except KeyError:
            logging.error("Turn trick cannot be applied without 'turn' parameters.")

        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                
                # if buffer is empty, encode the gradient
                if self._buffer_empty:
                    encodedTensor = self.grace.compress(param.grad.data, turn=self.turn)
                    self._gatheredGradients[i] += self.grace.decompress(encodedTensor, shape=param.grad.data.shape)
                # if buffer in nonempty, encode the residual
                elif self.grace._current_sign == 1:
                    encodedTensor = self.grace.compress_with_reference(param.grad, self._plus_sign_buffer[i])
                    self._gatheredGradients[i] += self.grace.decompress_with_reference(encodedTensor, self._plus_sign_buffer[i])
                else:
                    encodedTensor = self.grace.compress_with_reference(param.grad, self._minus_sign_buffer[i])
                    self._gatheredGradients[i] += self.grace.decompress_with_reference(encodedTensor, self._minus_sign_buffer[i])

                if self.grace._current_sign == 1:
                    self._minus_sign_buffer[i] += (param.grad.data < -const.EPSILON)
                else:
                    self._plus_sign_buffer[i] += (param.grad.data > const.EPSILON)

                # clear the gradients for next step, which is equivalent to zero_grad()
                param.grad.detach_()
                param.grad.zero_() 


    def step(self):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            for i, param in enumerate(group['params']):
                d_param = self.grace.transAggregation(self._gatheredGradients[i])

                # register buffer
                if self.grace._current_sign == 1:
                    self._plus_sign_buffer[i].zero_()
                    self._minus_sign_buffer[i] = self.grace.transAggregation(self._minus_sign_buffer[i])            
                else:
                    self._minus_sign_buffer[i].zero_()
                    self._plus_sign_buffer[i] = self.grace.transAggregation(self._plus_sign_buffer[i])
                
                param.data.add_(-group["lr"], d_param)
                self._gatheredGradients[i].zero_()
                
        self._buffer_empty = False


def grace_optimizer(optimizer, grace, **kwargs):
    """
    An optimizer that wraps another torch.optim.Optimizer.

    Allreduce operations are executed after each gradient is computed by ``loss.backward()``
    in parallel with each other. The ``step()`` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    Args:
        optimizer (torch.nn.optim.Optimizer):   Optimizer to use for computing gradients and applying updates.
        grace (grace_fl.Compressor):            Compression algorithm used during allreduce to reduce the amount
        mode (int):                             mode represents different implementations of optimizer.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method.

    """>>> TODO: add another 2 modes"""
    if "mode" in kwargs:
        mode = kwargs["mode"]
    else:
        mode = 3

    if mode==0:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
        dict(_predTurnOptimizer.__dict__))
    else:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
            dict(_graceOptimizer.__dict__))

    return cls(optimizer.param_groups, grace, **kwargs)
