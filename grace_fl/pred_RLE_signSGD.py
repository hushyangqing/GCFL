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
    runLengths = []
    
    iter = 0
    bitThreshold = 2**thres 
    for element in clonedTensor:
        iter += 1

        if element == currSymbol:
            runLength += 1
            if runLength >= bitThreshold - 1:
                codedSeqs.append(runLength)
                codedSeqs.append(currSymbol)
                runLengths.append(runLength)

                if iter != np.sum(runLengths):
                    print("Error")

                runLength = 0
        else:
            codedSeqs.append(runLength)
            codedSeqs.append(currSymbol)
            runLengths.append(runLength)
            
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
        self._require_grad_idx = True
        self.compress_ratios = []
        self._gradBuffer = []
        self._residualBuffer = []
        self._bufferEmpty = None
        self._code_dtype_bit = const.NIBBLE_BIT

    def register(self, model_params):
        """register the model parameters in the buffer.

        Args,
            model_params (list):  a list of the model parameters. 
        """
        for counter, param in enumerate(model_params):
            self._gradBuffer.append(param)
            self._residualBuffer.append(torch.zeros_like(param))
        
        counter += 1
        self._bufferEmpty = torch.ones(counter, dtype=torch.bool)

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor):  the input tensor.
            gradIdx (int):          the index of the gradient tensor wrt. the whole model.
        """
        encodedTensor = []
        try:
            gradIdx = kwargs["gradIdx"]
            turn = kwargs["turn"]
        except KeyError:
            logging.error("Cannot parse input for pred_signSGD compressor.")

        if turn%2 == 0:
            if self._bufferEmpty[gradIdx]:
                self._gradBuffer[gradIdx] = (tensor>0)
                encodedTensor = rl_enc(self._gradBuffer[gradIdx], thres=self._code_dtype_bit)
                self._bufferEmpty[gradIdx] = False
            else:
                self._residualBuffer[gradIdx] = ((tensor>0) != self._gradBuffer[gradIdx])
                encodedTensor = rl_enc(self._residualBuffer[gradIdx], thres=self._code_dtype_bit)
        else:
            if self._bufferEmpty[gradIdx]:
                self._gradBuffer[gradIdx] = (tensor<0)
                encodedTensor = rl_enc(self._gradBuffer[gradIdx], thres=self._code_dtype_bit)
                self._bufferEmpty[gradIdx] = False
            else:
                self._residualBuffer[gradIdx] = ((tensor<0) != self._gradBuffer[gradIdx])
                encodedTensor = rl_enc(self._residualBuffer[gradIdx], thres=self._code_dtype_bit)

        rawBits = torch.prod(torch.tensor(tensor.shape)) * const.FLOAT_BIT
        codedBits = torch.tensor(len(encodedTensor) * self._code_dtype_bit, dtype=torch.float)
        
        return encodedTensor

    def decompress(self, codes, shape):
        """Decoding the tensor codes to float format."""
        decodedTensor = rl_dec(codes)
        decodedTensor = decodedTensor.view(shape)

        return decodedTensor
            

    def compressRatio(self):
        return np.mean(np.asarray(self.compress_ratios))

    def transAggregation(self, tensor, **kwargs):
        """Transform a raw aggregation sum. 

        Args,
            tensor (torch.Tensor): the input aggregation tensor.
        """
        try:
            assert("turn" in kwargs)
        except AssertionError:
            logging.info("pred_RLE_SignSGD needs turn as a parameter.")
        
        if kwargs["turn"] % 2 == 0:
            sign = -1
        else:
            sign = 1

        onesTensor = torch.ones_like(tensor)
        aggedTensor = torch.where(tensor >=0, onesTensor, -onesTensor)
        return sign*aggedTensor

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
