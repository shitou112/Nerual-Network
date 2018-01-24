import numpy as np

class Relu(object):

    def __call__(self, x):
        return np.where(x>0, x, 0)

    def grad(self, x):
        return np.where(x>0, x, 0)