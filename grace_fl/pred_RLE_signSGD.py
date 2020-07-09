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
    
    iteration = 0
    bitThreshold = 2**thres 
    for element in clonedTensor:
        iteration += 1

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
        for i in range(runLength):
            decodedTensor.append(symbol)

    decodedTensor = torch.tensor(decodedTensor)
    decodedTensor = decodedTensor.to(torch.float32)
    decodedTensor = decodedTensor.to(codedSeqs.device)
    decodedTensor = decodedTensor.flatten()

    return decodedTensor
    

class PredRLESignSGDCompressor(Compressor):

    def __init__(self, thres):
        super().__init__()
        self._const_compress_ratio = False
        self._code_dtype_bit = const.WORD_BIT
        self.th = thres

        # total number of symbols (gradient coordinates) & number of residuals
        self.total_symbols = 0
        self.residual_symbols = 0

    def compress(self, tensor, sign):
        """
        Compress the input tensor with run-length of sign and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor):  the input tensor.
            sign (int):             1 for "+" and -1 for "-".
        """
        if sign == 1:
            signs = (tensor > const.EPSILON)
        else:
            signs = (tensor < -const.EPSILON)
            
        encodedTensor = rl_enc(signs, thres=self._code_dtype_bit)

        rawBits = torch.prod(torch.tensor(tensor.shape)) * const.FLOAT_BIT
        codedBits = torch.tensor(len(encodedTensor) * self._code_dtype_bit, dtype=torch.float)
        self.compress_ratios.append(rawBits/codedBits)

        return encodedTensor

    def compress_with_reference(self, tensor, refTensor, sign):
        """
        Given a reference sign tensor, compress the residual between the input tensor and the reference with run-length encoding.

        Args,
            tensor (torch.tensor):  the input tensor.
            refTensor (torch.tensor): the reference tensor.
            sign (int):             1 for "+" and -1 for "-"
        """

        if sign == 1:
            residual = ((tensor > 0) != refTensor)
        else:
            residual = ((tensor < 0) != refTensor)

        encodedTensor = rl_enc(residual, thres=self._code_dtype_bit)

        return encodedTensor

    def decompress(self, codes, shape):
        """Decode the tensor codes to float format."""
        decodedTensor = rl_dec(codes)
        decodedTensor = decodedTensor.view(shape)
        decodedTensor = decodedTensor
        return decodedTensor

    def decompress_with_reference(self, tensor, refTensor):
        """Decode the residual tensor given the reference tensor.
        
        Args:
            tensor (torch.tensor, bool):    the residual tensor.
            refTensor (torch.tensor, bool): the reference tensor.
        """
        decodedTensor = rl_dec(tensor)
        decodedTensor = decodedTensor.view_as(refTensor)
        decodedTensor = torch.where(decodedTensor==1,1-refTensor, refTensor)
        decodedTensor = decodedTensor.to(torch.float32)

        return decodedTensor

    @property
    def compress_ratio(self):
        """Use the entropy as a compression estimation."""
        residual_ratio = self.residual_symbols / self.total_symbols
        entropy = -residual_ratio*np.log2(residual_ratio) - (1-residual_ratio)*np.log2(1-residual_ratio)

        return const.FLOAT_BIT/entropy

    def reset(self):
        self.total_symbols = 0
        self.residual_symbols = 0

    def trans_aggregation(self, tensor):
        """Transform a raw aggregation sum. 

        Args,
            tensor (torch.Tensor): the input aggregation tensor.
        """

        onesTensor = torch.ones_like(tensor)
        zerosTensor = torch.zeros_like(tensor)
        aggedTensor = torch.where(tensor > self.th, onesTensor, zerosTensor)
        return aggedTensor

    def aggregate(self, tensors):
        """Aggregate a list of tensors.
        
        Args,
            tensors (torch.Tensor): `tensors` have more than three dimensions, which all of 
                                    the candidates concantented in the first channel.  
        """
        
        aggedTensor = sum(tensors)
        onesTensor = torch.ones_like(tensor)
        zerosTensor = torch.zeros_like(tensors)
        aggedTensor = torch.where(aggedTensor > self.th, onesTensor, zerosTensor)
        return aggedTensor
