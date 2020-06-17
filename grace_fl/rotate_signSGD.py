import torch

# My libraries
from grace_fl import Compressor
import grace_fl.constant as const 

class RotateSignSGDCompressor(Compressor):

    def __init__(self):
        super().__init__()
        self.dtype = torch.uint8
        self._const_compress_ratio = None

    def compress(self, tensor):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        encodedTensor = (tensor >= 0)
        return encodedTensor

    def decompress(self, tensors, shape):
        """Decoding the signs to float format """
        DecoedTensor = tensors.type(torch.float32) * 2 - 1
        DecoedTensor = DecoedTensor.view(shape)
        return DecoedTensor

    def compressRatio(self):
        return self._const_compress_ratio

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
