import numpy as np
import logging

# PyTorch Libraries
import torch

# My libraries
from grace_fl import Compressor
import grace_fl.constant as const 

class PredSignSGDCompressor(Compressor):
    def __init__(self):
        super().__init__()
        self._current_sign = 1
        self.dtype = torch.uint8
        self._const_compress_ratio = const.FLOAT_BIT / const.BINARY_BIT
        self.compress_ratios = []

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with run-length of sign and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor):  the input tensor.
            turn (int):             odd or even t.
        """

        # even turn send the "+" and odd turn send the "-""
        if self._current_sign == 1:
            signs = (tensor > const.EPSILON)
        else:
            signs = (tensor < -const.EPSILON)

        encodedTensor = signs
        return encodedTensor

    def compress_with_reference(self, tensor, refTensor):
        """
        Given a reference sign tensor, compress the residual between the input tensor and the reference with run-length encoding.

        Args,
            tensor (torch.tensor):  the input tensor.
            refTensor (torch.tensor): the reference tensor.
        """
        if self._current_sign == 1:
            residual = (tensor > const.EPSILON) != refTensor
        else:
            residual = (tensor < -const.EPSILON) != refTensor  

        encodedTensor = residual
        return encodedTensor
    
    @property
    def compressRatio(self):
        return self._const_compress_ratio

    def decompress(self, codes, shape):
        """Decode the tensor codes to float format."""
        decodedTensor = codes.to(torch.float32)
        decodedTensor = decodedTensor.view(shape)
        decodedTensor = self._current_sign * decodedTensor
        return decodedTensor

    def decompress_with_reference(self, tensor, refTensor):
        """Decode the residual tensor given the reference tensor.
        
        Args:
            tensor (torch.tensor):    the residual tensor.
            refTensor (torch.tensor): the reference tensor.
        """
        decodedTensor = torch.where(tensor==1, 1-refTensor, refTensor)
        decodedTensor = decodedTensor.to(torch.float32)
        decodedTensor = self._current_sign * decodedTensor

        return decodedTensor

    def compressRatio(self):
        return self._const_compress_ratio

    def transAggregation(self, tensor):
        """Transform a raw aggregation sum. 

        Args,
            tensor (torch.Tensor): the input aggregation tensor.
        """

        onesTensor = torch.ones_like(tensor)
        zerosTensor = torch.zeros_like(tensor)

        if self._current_sign == 1:
            aggedTensor = torch.where(tensor > 0, onesTensor, zerosTensor)
        else:
            aggedTensor = torch.where(tensor < 0, -onesTensor, zerosTensor)
        return aggedTensor

