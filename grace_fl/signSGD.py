import torch

# My libraries
from grace_fl import Compressor
import grace_fl.constant as const 

class SignSGDCompressor(Compressor):

    def __init__(self):
        super().__init__()
        self.dtype = torch.uint8

    def compress(self, tensor):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        encodedTensor = (tensor >= 0)
        shapeProduct = torch.prod(tensor.shape)
        rawBits = const.FLOAT_TENSOR_BIT * shapeProduct
        encodedBits = const.BINARY_TENSOR_BIT * shapeProduct
        
        return dict(encodedTensor = encodedTensor,
                    rawBits = rawBits,
                    encodedBits = encodedBits
               )

    def decompress(self, tensors, shape):
        """Decoding the signs to float format """
        DecoedTensor = tensors.type(torch.float32) * 2 - 1
        DecoedTensor = DecoedTensor.view(shape)
        return DecoedTensor

    def transAggregation(self, tensor):
        """Transform a raw aggregation sum. 

        Args,
            tensor (torch.Tensor): the input aggregation tensor.
        """
        aggedTensor = torch.where(tensor >=0, torch.tensor(1.), torch.tensor(-1.))
        return aggedTensor

    def aggregate(self, tensors):
        """Aggregate a list of tensors.
        
        Args,
            tensors (torch.Tensor): `tensors` have more than three dimensions, which all of 
                                    the candidates concantented in the first channel.  
        """
        
        aggedTensor = sum(tensors)
        aggedTensor = torch.where(aggedTensor >=0, torch.tensor(1.), torch.tensor(-1.))
        return aggedTensor
