import numpy as np
import logging

# PyTorch Libraries
import torch

# My libraries
from grace_fl import Compressor
import grace_fl.constant as const 

def rl_enc(tensor, thres=const.NIBBLE_BIT):
    """run-length encoding for a binary tensor. 
    """
    clonedTensor = tensor.flatten().clone()
    currSymbol = 0 if clonedTensor[0] == 0 else 1 
    codedSeqs = []
    runLength = 0
    
    iter = 0
    bitThreshold = 2**thres 
    for element in clonedTensor:
        iter += 1

        if element == currSymbol:
            runLength += 1
            if runLength >= bitThreshold - 1:
                codedSeqs.append(runLength)
                codedSeqs.append(currSymbol)

                runLength = 0
        else:
            codedSeqs.append(runLength)
            codedSeqs.append(currSymbol)
            
            runLength = 1
            currSymbol = 0 if element == 0 else 1 

    # Last part of the code sequence
    codedSeqs.append(runLength)
    codedSeqs.append(currSymbol)
    
    codedSeqs = torch.from_numpy(np.asarray(codedSeqs))
    codedSeqs = codedSeqs.to(tensor.device)
    return codedSeqs

def rl_dec(codedSeqs):
    """run-length decoding from a code sequence. The first element is set to zero by force.
    """
    
    decodedTensor = []
    step = 2
    symbols = [codedSeqs[i] for i in range(1, len(codedSeqs), step)]
    runLengths = [codedSeqs[i] for i in range(0, len(codedSeqs), step)]
    
    for symbol, runLength in zip(symbols, runLengths):
        for iter in range(runLength):
            decodedTensor.append(symbol)

    decodedTensor = torch.tensor(decodedTensor)
    decodedTensor = decodedTensor.to(torch.float32)
    decodedTensor = decodedTensor.to(codedSeqs.device)
    decodedTensor = decodedTensor.flatten()

    return decodedTensor
    

class PredRLESignSGDCompressor(Compressor):

    def __init__(self):
        super().__init__()
        self.dtype = torch.uint8
        self._const_compress_ratio = False
        self.compress_ratios = []
        self._code_dtype_bit = const.NIBBLE_BIT

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with run-length of sign and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor):  the input tensor.
            turn (int):             odd or even t.
        """
        try:
            turn = kwargs["turn"]
        except KeyError:
            logging.error("Cannot parse input for pred_signSGD compressor.")

        # even turn send the "+" and odd turn send the "-""
        if turn%2 == 0:
            signs = (tensor > const.EPSILON)
            self._current_sign = 1
        else:
            signs = (tensor < const.EPSILON)
            self._current_sign = -1

        encodedTensor = rl_enc(signs)

        rawBits = torch.prod(torch.tensor(tensor.shape)) * const.FLOAT_BIT
        codedBits = torch.tensor(len(encodedTensor) * self._code_dtype_bit, dtype=torch.float)
        self.compress_ratios.append(rawBits/codedBits)

        return encodedTensor

    def compress_with_reference(self, tensor, refTensor):
        """
        Given a reference sign tensor, compress the residual between the input tensor and the reference with run-length encoding.

        Args,
            tensor (torch.tensor):  the input tensor.
            refTensor (torch.tensor): the reference tensor.
        """
        if self._current_sign == 1:
            residual = (tensor > 0) != refTensor
        else:
            residual = (tensor < 0) != refTensor  

        encodedTensor = rl_enc(residual)

        rawBits = torch.prod(torch.tensor(tensor.shape)) * const.FLOAT_BIT
        codedBits = torch.tensor(len(encodedTensor) * self._code_dtype_bit, dtype=torch.float)
        self.compress_ratios.append(rawBits/codedBits)

        return encodedTensor

    def decompress(self, codes, shape):
        """Decoding the tensor codes to float format."""
        decodedTensor = rl_dec(codes)
        decodedTensor = decodedTensor.view(shape)
        decodedTensor = self._current_sign * decodedTensor
        return decodedTensor

    def compressRatio(self):
        """Take the average of compress ratio array as an estimation."""
        return np.mean(np.asarray(self.compress_ratios))

    def transAggregation(self, tensor):
        """Transform a raw aggregation sum. 

        Args,
            tensor (torch.Tensor): the input aggregation tensor.
        """

        onesTensor = torch.ones_like(tensor)
        zerosTensor = torch.zeros_like(tensor)
        aggedTensor = torch.where(tensor > 0, onesTensor, zerosTensor)
        aggedTensor = self._current_sign * aggedTensor
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
