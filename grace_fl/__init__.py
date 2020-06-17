"""
Refer to grace: https://github.com/sands-lab/grace
"""
from abc import ABC, abstractmethod

class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def register(self, ctx):
        """register the model information in the model-related compressor"""

    @abstractmethod
    def compress(self, tensor, compress_ctx):
        """Compresses a tensor with the given compression context, and then returns it with the context needed to decompress it."""

    @abstractmethod
    def decompress(self, tensors, decompress_ctx):
        """Decompress the tensor with the given decompression context."""

    def transAggregation(self, tensor):
        """Transform a raw aggregation sum. """

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)


from grace_fl.signSGD import SignSGDCompressor

compressorRegistry = {
"signSGD": SignSGDCompressor
}
