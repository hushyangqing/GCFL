import numpy as np
import logging

# PyTorch Libraries
import torch

# My libraries
from grace_fl import Compressor
import grace_fl.constant as const 

class IdealBinaryPredSignSGDCompressor(Compressor):
    def __init__(self, config):
        super().__init__()
        self.dtype = torch.uint8
        self.majority_thres = int(0.5 * config.users * config.sampling_fraction)
        self._const_compress_ratio = False
        self.compress_ratios = []

    def compress(self, tensor):
        """
        Compress the input tensor with run-length of sign and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor):  the input tensor.
        """
        ones_tensor = torch.ones_like(tensor)
        zeros_tensor = torch.zeros_like(tensor)
        encodedTensor = torch.where(tensor>0, ones_tensor, zeros_tensor)
        return encodedTensor

    def compress_with_reference(self, tensor, ref_tensor):
        """
        Given a reference tensor, compress the residual between the input tensor and the reference.
        An ideal compressor supposes that the ratio is equal to entropy.

        Args,
            tensor (torch.tensor):  the input tensor.
            ref_tensor (torch.tensor): the reference tensor.
        """
        residual = ((tensor > 0) != ref_tensor)
        encodedTensor = residual

        num_elements = torch.sum(residual)
        ratio_elements = num_elements / np.prod(residual.shape)
        entropy = -(ratio_elements)*np.log2(ratio_elements) - (1-ratio_elements)*np.log2(1-ratio_elements)

        self.compress_ratios.append(entropy)

        return encodedTensor

    def decompress(self, codes, shape):
        """Decode the tensor codes to float format."""
        decoded_tensor = codes.to(torch.float32)
        decoded_tensor = decoded_tensor.view(shape)
        decoded_tensor = 2*decoded_tensor - 1
        return decoded_tensor

    def decompress_with_reference(self, tensor, ref_tensor):
        """Decode the residual tensor given the reference tensor.
        
        Args:
            tensor (torch.tensor):    the residual tensor.
            ref_tensor (torch.tensor): the reference tensor.
        """
        decoded_tensor = torch.where(tensor==1, 1-ref_tensor, ref_tensor)
        decoded_tensor = decoded_tensor.to(torch.float32)
        decoded_tensor = 2*decoded_tensor - 1

        return decoded_tensor

    @property
    def compress_ratio(self):
        """Take the average of compress ratio array as an estimation."""
        return np.mean(np.asarray(self.compress_ratios))

    def reset(self):
        self.compress_ratios = []

    def trans_aggregation(self, tensor):
        """Transform a raw aggregation sum. 

        Args,
            tensor (torch.Tensor): the input aggregation tensor.
        """

        onesTensor = torch.ones_like(tensor)
        aggedTensor = torch.where(tensor > 0, onesTensor, -onesTensor)
        return aggedTensor

