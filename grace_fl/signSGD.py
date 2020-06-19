import torch

# My libraries
from grace_fl import Compressor
import grace_fl.constant as const 

class SignSGDCompressor(Compressor):

    def __init__(self):
        super().__init__()
        self.dtype = torch.uint8
        self._const_compress_ratio = const.FLOAT_BIT / const.BINARY_BIT

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        encodedTensor = (tensor >= 0)
        return encodedTensor

    def decompress(self, tensors, shape):
        """Decoding the signs to float format """
        decodedTensor = tensors.type(torch.float32) * 2 - 1
        decodedTensor = decodedTensor.view(shape)
        return decodedTensor

    def compressRatio(self):
        return self._const_compress_ratio

    def transAggregation(self, tensor, **kwargs):
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
