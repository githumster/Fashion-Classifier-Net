import numpy as np
from .base import Criterion

'''
The MSECriterion, which is basic L2 norm usually used for regression

- input: batch_size x n_feats
- target: batch_size x n_feats
- output: scalar
'''
class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def updateOutput(self, input, target):   
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output 
 
    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"
    

'''
- input: batch_size x n_feats - probabilities
- target: batch_size x n_feats - one-hot representation of ground truth
- output: scalar
'''
class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    
    def __init__(self):
        super(ClassNLLCriterionUnstable, self).__init__()
        
    def updateOutput(self, input, target): 
        # Trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        
        # Convert one-hot encoded target to class indices
        if target.ndim > 1:
            target = np.argmax(target, axis=1)
        
        self.output = -np.mean(np.log(input_clamp[np.arange(len(target)), target]))
        return self.output
    
    def updateGradInput(self, input, target):
        # Trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        
        # Convert one-hot encoded target to class indices 
        if target.ndim > 1:
            target = np.argmax(target, axis=1)
        
        self.gradInput = np.zeros_like(input)
        
        # Calculate gradient with respect to input
        N = input.shape[0]
        self.gradInput[np.arange(N), target] = -1 / input_clamp[np.arange(N), target]
        self.gradInput /= N
        
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterionUnstable"




class ClassNLLCriterion(Criterion):
    def __init__(self):
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        # Convert one-hot encoded target to class indices 
        if target.ndim > 1:
            target = np.argmax(target, axis=1)

        # Clamp the values to avoid numerical instability
        input_clamp = np.clip(input, 1e-15, 1 - 1e-15)
        
        # Calculate loss for each sample
        batch_size = input.shape[0]
        loss = -np.sum(input[np.arange(batch_size), target])
        self.output = loss / batch_size
        return self.output
    
    def updateGradInput(self, input, target):
        # Convert one-hot encoded target to class indices
        if target.ndim > 1:
            target = np.argmax(target, axis=1)

        self.gradInput = np.zeros_like(input)
        input_clamp = np.clip(input, 1e-15, 1 - 1e-15)
        batch_size = input.shape[0]
        
        # Gradient calculation
        self.gradInput[np.arange(batch_size), target] = -1
        self.gradInput = self.gradInput / batch_size
        
        return self.gradInput
    
    def __repr__(self):
        return "ClassNLLCriterion"