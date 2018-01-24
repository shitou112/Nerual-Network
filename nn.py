

class NerualNetwork(object):

    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def _forword_pass(self, X):
        for layer in self.layers:
            outputs = layer.forward_pass(X)

        return outputs

    def _backword_pass(self, acc_grad):
        for layer in reversed(self.layers):
            layer.backword_pass(acc_grad)



    def fit(self, X, y, epochs=5):
        for i in range(epochs):
            pred = self._forword_pass(X)
            loss = self.loss.acc(pred, y)
            loss_grad = self.loss.grad(pred, y)
            self._backword_pass(loss_grad)

    def predict(self, X):