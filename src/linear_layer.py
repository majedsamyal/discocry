import numpy as np

class layer:
    def __init__(self,input_feature,output_feature):
        self.W = np.random.randn(output_feature,input_feature) * 0.01
        self.b = np.zeros((output_feature, 1)) # Always single vector


    def forward(self,x):
        # Forward pass , operate on base values to find the error/loss.
        # Here x is the feature matric which we will transform
        self.input = x
        # Now find outut using output = W * X + b 
        self.output = np.dot(self.W,x) + self.b 
        return self.output

        
    def backward(self,grad_output, learing_rate):
        # Find which Weight to blame
        # We can't change the magnitude of inputs but with Weight adjustment we can control the impact.
        batch_size = self.input.shape[1]
        # This calculates that blame for EVERY weight grad_W = error × input
        self.grad_W = np.dot(grad_output,self.input.T) / batch_size
        self.grad_b = np.sum(grad_output, axis=1, keepdims=True) / batch_size
        grad_input = np.dot(self.W, grad_output)
        # update weight and biase now 
        self.W = learing_rate * self.grad_W
        self.b = learing_rate * self.grad_b

        return grad_input # pass back feature blame


