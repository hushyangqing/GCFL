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
        
        # total number of symbols (gradient coordinates) & number of residuals
        self.total_symbols = 0
        self.residual_symbols = 0

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
        ones_tensor = torch.ones_like(tensor)
        sign_tensor = torch.where(tensor>0, ones_tensor, -ones_tensor)
        residual = sign_tensor - ref_tensor
        encodedTensor = residual

        self.total_symbols += np.prod(residual.shape)
        self.residual_symbols += torch.sum(residual != 0).item()

        return encodedTensor

    def decompress(self, codes, shape):
        """Decode the tensor codes to float format."""
        decoded_tensor = codes.to(torch.float32)
        decoded_tensor = decoded_tensor.view(shape)
        decoded_tensor = 2*decoded_tensor - 1
        return decoded_tensor

    def decompress_with_reference(self, tensor, ref_tensor):
        """Decode the tensor given the reference tensor.
        
        Args:
            tensor (torch.tensor):    the residual tensor.
            ref_tensor (torch.tensor): the reference tensor.
        """
        decoded_tensor = tensor + ref_tensor

        return decoded_tensor

    @property
    def compress_ratio(self):
        """Use the entropy as a compression estimation."""
        residual_ratio = self.residual_symbols / self.total_symbols
        if residual_ratio == 1:
            return const.FLOAT_BIT/const.BINARY_BIT
        else:
            entropy = -residual_ratio*np.log2(residual_ratio) - (1-residual_ratio)*np.log2(1-residual_ratio)
            print("entropy: {:.3f}".format(entropy))
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
        aggedTensor = torch.where(tensor > 0, onesTensor, -onesTensor)
        return aggedTensor

