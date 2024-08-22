import numpy as np
from .base import Module


'''
- input: batch_size x n_feats
- output: batch_size x n_feats
'''
class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        expa = np.exp(self.output)
        self.output = expa / expa.sum(axis=1,keepdims=True)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        softmax = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
        softmax_diag = np.einsum('ij,jk->ijk', softmax, np.eye(softmax.shape[1]))
        softmax_outer = np.einsum('ij,ik->ijk', softmax, softmax)
        jacobian = softmax_diag - softmax_outer
        gradInput = np.einsum('ijk,ij->ik', jacobian, gradOutput)
        return gradInput
    
    def __repr__(self):
        return "SoftMax"
    
'''
- input: batch_size x n_feats
- output: batch_size x n_feats
'''
class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        expa = np.exp(self.output)
        self.output = np.log(expa / expa.sum(axis=1,keepdims=True))
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        stabilized_input = input - input.max(axis=1, keepdims=True)
        softmax = np.exp(stabilized_input) / np.sum(np.exp(stabilized_input), axis=1, keepdims=True)
        sum_grad = np.sum(gradOutput, axis=1, keepdims=True)
        self.gradInput = gradOutput - softmax * sum_grad
        
        return self.gradInput
    def __repr__(self):
        return "LogSoftMax"


class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput
    
    def __repr__(self):
        return "ReLU"
    


class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def updateOutput(self, input):
        self.output = np.where(input >= input * self.slope, input, input * self.slope)
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input >= input * self.slope, 1, self.slope) * gradOutput
        return self.gradInput
    
    def __repr__(self):
        return "LeakyReLU"
    


class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        
    def updateOutput(self, input):
        self.output = np.where(input >= 0, input, self.alpha * (np.exp(input) - 1)) 
        return  self.output
    
    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input >= 0, 1, self.alpha * np.exp(input)) * gradOutput
        return self.gradInput
    
    def __repr__(self):
        return "ELU"
    



class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def updateOutput(self, input):
        self.output = np.log(1 + np.exp(input))
        return  self.output
    def updateGradInput(self, input, gradOutput):
        self.gradInput = 1 / (1 + np.exp(input)) * np.exp(input) * gradOutput
        return self.gradInput
    
    def __repr__(self):
        return "SoftPlus"
    