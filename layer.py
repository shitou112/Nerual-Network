import numpy as np

class Layer(object):

    def __init__(self):
        pass

    def forward_pass(self, X):
        raise NotImplementedError()

    def backword_pass(self, acc_grad):
        raise NotImplementedError()




class Dense(Layer):

    def __init__(self, inputs, n_neruals):
        self.inputs = inputs
        self.n_neruals = n_neruals

    def initialize(self, optimizer, mean=0, sigma=1):
        self.W = np.random.normal(loc=mean, scale=sigma, size=(self.inputs, self.n_neruals))
        self.b = np.zeros((self.n_neruals, ))
        self.optimizer = optimizer

    def forward_pass(self, X):
        self.inputs = X
        outputs = X.dot(self.W) + self.b

        return outputs

    def backword_pass(self, acc_grad):
        grad_w = self.W, self.inputs.T.dot(acc_grad)
        self.W = self.optimizer.update(grad_w)
        grad_b = acc_grad.sum(axis=0, keep_dim=True)
        self.b = self.optimizer.update(self.b, grad_b)

        return self.inputs.dot(acc_grad)



class relu(Layer):

    def forward_pass(self, X):
        self.inputs = X
        return np.where(X>=0, X, 0)

    def backword_pass(self, acc_grad):
        return acc_grad*np.where(self.inputs>=0, 1, 0)