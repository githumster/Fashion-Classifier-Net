import numpy as np

from .base import Module


'''
Also known as dense layer, fully-connected layer, FC-layer, InnerProductLayer (in caffe), affine transform
- input: batch_size x n_feats1
- output: batch_size x n_feats2
'''
class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def updateOutput(self, input):
        
        self.output = input@self.W.T + self.b
        return self.output
    
    def updateGradInput(self, input, gradOutput):  
        self.gradInput = gradOutput @ self.W
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        
        self.gradW = gradOutput.T @ input
        self.gradb = gradOutput.sum(axis=0)
        pass
    
    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def getParameters(self):
        return [self.W, self.b]
    
    def getGradParameters(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q

'''
- input: batch_size x n_feats
- output: batch_size x n_feats

The layer should work as follows. While training (self.training == True) it transforms input as 
y = \frac{x - \mu} {\sqrt{\sigma + \epsilon}}
where mu and sigma - mean and variance of feature values in batch and \epsilon is just a small number for numericall stability. 
Also during training, layer should maintain exponential moving average values for mean and variance: 

self.moving_mean = self.moving_mean * alpha + batch_mean * (1 - alpha)
self.moving_variance = self.moving_variance * alpha + batch_variance * (1 - alpha)

During testing (`self.training == False`) the layer normalizes input using moving_mean and moving_variance. 

Note that decomposition of batch normalization on normalization itself and channelwise scaling here is just a common implementation choice. 
In general "batch normalization" always assumes normalization + scaling.
'''

class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None 
        self.moving_variance = None
        
    def updateOutput(self, input):
        if self.training:
            # Calculate batch mean and variance
            batch_mean = np.mean(input, axis=0, keepdims=True)
            batch_variance = np.var(input, axis=0, keepdims=True)

            # Normalize the input
            self.output = (input - batch_mean) / np.sqrt(batch_variance + self.EPS)

            # Update moving mean and variance
            if self.moving_mean is None:
                self.moving_mean = batch_mean
                self.moving_variance = batch_variance
            else:
                self.moving_mean = self.moving_mean * self.alpha + batch_mean * (1 - self.alpha)
                self.moving_variance = self.moving_variance * self.alpha + batch_variance * (1 - self.alpha)
        else:
            # Use moving mean and variance for normalization during testing
            self.output = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        
        return self.output
    
    def updateGradInput(self, input, gradOutput):
        N, D = input.shape

        # Calculate gradients
        batch_mean = np.mean(input, axis=0, keepdims=True)
        batch_variance = np.var(input, axis=0, keepdims=True)
        std_inv = 1. / np.sqrt(batch_variance + self.EPS)

        x_mu = input - batch_mean
        grad_var = np.sum(gradOutput * x_mu * -0.5 * std_inv**3, axis=0)
        grad_mean = np.sum(gradOutput * -std_inv, axis=0) + grad_var * np.mean(-2. * x_mu, axis=0)

        self.gradInput = (gradOutput * std_inv) + (grad_var * 2 * x_mu / N) + (grad_mean / N)
        return self.gradInput
    
    def __repr__(self):
        return "BatchNormalization"
    


class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output
        
    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput
    
    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)
    
    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def getParameters(self):
        return [self.gamma, self.beta]
    
    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"
    
'''

While training (self.training == True) it should sample a mask on each iteration (for every batch), zero out elements and 
multiply elements by $1 / (1 - p)$. The latter is needed for keeping mean values of features close to mean values which will be in test mode. 
When testing this module should implement identity transform i.e. self.output = input.

- input:   **`batch_size x n_feats`**
- output: **`batch_size x n_feats`**
'''
class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = None
        
    def updateOutput(self, input):
    
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape)
            self.output = input * self.mask * (1/(1 - self.p))
        else:
            self.output = input
        return self.output
  
    
    def updateGradInput(self, input, gradOutput):
        if self.training:
            # The gradient is transmitted only through "on" neurons
            self.gradInput = gradOutput * self.mask / (1 - self.p)
        else:
            # While training gradient no changes
            self.gradInput = gradOutput
        return self.gradInput
        
    def __repr__(self):
        return "Dropout"
    



# Define a forward and backward pass procedures.
class Sequential(Module):
    """
    This class implements a container that processes input data sequentially. 
    input is processed by each module (layer) in self.  modules consecutively.
    The resulting array is called output. 
    """
    
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.modules = list(modules) 
        
    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:
        """
        self.input = input  # Store input for backward pass
        self.outputs = []  # To store intermediate outputs
        ans = input
        for module in self.modules:
            ans = module.forward(ans)
            self.outputs.append(ans)  # Store the output of each module
        self.output = ans
        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:
        """
        gradInput = gradOutput
        # Traverse in reverse order
        for i in reversed(range(len(self.modules))):
            module = self.modules[i]
            output = self.outputs[i]
            gradInput = module.backward(self.input if i == 0 else self.outputs[i-1], gradInput)
        self.gradInput = gradInput
        return self.gradInput
    
    def zeroGradParameters(self): 
        """
        Zero out all gradients for all parameters in the modules.
        """
        for module in self.modules:
            module.zeroGradParameters()
    
    def getParameters(self):
        """
        Gather all parameters from all modules.
        """
        params = []
        for module in self.modules:
            params.extend(module.getParameters())
        return params
    
    def getGradParameters(self):
        """
        Gather all gradients w.r.t parameters from all modules.
        """
        grads = []
        for module in self.modules:
            module_grads = module.getGradParameters()
            if isinstance(module_grads, list) or isinstance(module_grads, tuple):
                grads.extend(module_grads)
            else:
                grads.append(module_grads)
        return grads

    def __repr__(self):
        """
        Return a string representation of the sequential container.
        """
        string = "\n".join([str(x) for x in self.modules])
        return string
    
    def __getitem__(self, x):
        """
        Get the module at index `x`.
        """
        return self.modules.__getitem__(x)
    
    def train(self):
        """
        Set all modules to training mode.
        """
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        """
        Set all modules to evaluation mode.
        """
        self.training = False
        for module in self.modules:
            module.evaluate()
