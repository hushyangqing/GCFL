import numpy as np
import logging

# PyTorch Libraries
import torch

# My libraries
from grace_fl import Compressor
import grace_fl.constant as const 

def rl_enc(tensor):
    """run-length encoding for a binary tensor.
    """
    tensor_clone = tensor.flatten().clone()
    currSymbol = tensor_clone[0]
    codedSeries = [currSymbol]
    runLength = 1
    for element in tensor_clone[1:]:
        if element == currSymbol:
            runLength += 1
        else:
            codedSeries.append(codedSeries)
            runLength = 1

    return codedSeries

class PredRLESignSGDCompressor(Compressor):

    def __init__(self):
        super().__init__()
        self.dtype = torch.uint8
        self._const_compress_ratio = []
        self._gradBuffer = []
        self._residualBuffer = []
        self._bufferEmpty = None
        self.compress_ratios = []

    def register(self, model_params):
        """register the model parameters in the buffer.

        Args,
            model_params (list):  a list of the model parameters. 
        """
        for counter, param in enumerate(model_params):
            self._gradBuffer.append(param)
            self._residualBuffer.append(torch.zeros_like(param))
        
        counter += 1
        self._bufferEmpty = torch.zeros(counter, dtype=torch.bool)

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor):  the input tensor.
            grad (torch.tensor):    the current gradient to be compressed.
            gradIdx (int):          the index of the gradient tensor wrt. the whole model.
        """
        encodedTensor = []
        try:
            grad = kwargs["grad"]
            gradIdx = kwargs["gradIdx"]
        except KeyError:
            logging.error("Cannot parse input for pred_signSGD compressor.")

        if self._bufferEmpty[gradIdx]:
            self._gradBuffer[gradIdx] = grad
        else:
            self._residualBuffer[gradIdx] = (grad != self._gradBuffer[i])
            encodedTensor.append(rl_enc(self._residualBuffer[i]))

        return encodedTensor

    def decompress(self, codes, shape):
        """Decoding the tensor codes to float format."""
        tensorLength = torch.prod(torch.tensor(shape))
        decodedTensor = torch.zeros(tensorLength)
        zeroTensor = torch.tensor(0.)
        oneTensor = torch.tensor(1.)
        pointer = 0
        element = zeroTensor
        for code in codes:
            decodedTensor[pointer:pointer+code] = element
            # flip the element
            if element == zeroTensor:
                element = oneTensor
            else:
                element = zeroTensor

        decodedTensor = decodedTensor.view(shape)

        return decodedTensor
            

    def compressRatio(self):
        return np.mean(np.asarray(self.compress_ratios))

    def transAggregation(self, tensor):
        """Transform a raw aggregation sum. 

        Args,
            tensor (torch.Tensor): the input aggregation tensor.
        """
        onesTensor = torch.ones_like(tensor)
        aggedTensor = torch.where(tensor >=0, onesTensor, -onesTensor)
        return aggedTensor

    def aggregate(self, tensors):
        """Aggregate a list of tensors.
        
        Args,
            tensors (torch.Tensor): `tensors` have more than three dimensions, which all of 
                                    the candidates concantented in the first channel.  
        """
        
        aggedTensor = sum(tensors)
        onesTensor = torch.ones_like(tensor)
        aggedTensor = torch.where(aggedTensor >=0, onesTensor, -onesTensor)
        return aggedTensor
