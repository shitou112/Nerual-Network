
class SGD(object):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, w, acc_grad):
        return self.w - acc_grad * self.learning_rate
