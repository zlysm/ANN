import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        """The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation"""

        self._saved_tensor = tensor


class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        self._saved_for_backward(input)
        return np.where(
            input > 0, 1.0507 * input, 1.0507 * 1.67326 * (np.exp(input) - 1)
        )

    def backward(self, grad_output):
        input = self._saved_tensor
        return grad_output * np.where(
            input > 0, 1.0507, 1.0507 * 1.67326 * np.exp(input)
        )


class HardSwish(Layer):
    def __init__(self, name):
        super(HardSwish, self).__init__(name)

    def forward(self, input):
        self._saved_for_backward(input)

        return np.where(
            input <= -3, 0, np.where(input >= 3, input, input * (input + 3) / 6)
        )

    def backward(self, grad_output):
        input = self._saved_tensor
        return grad_output * np.where(
            input <= -3, 0, np.where(input >= 3, 1, input / 3 + 0.5)
        )


class Tanh(Layer):
    def __init__(self, name):
        super(Tanh, self).__init__(name)

    def forward(self, input):
        y = 1 - 2 / (np.exp(2 * input) + 1)
        self._saved_for_backward(y)
        return y

    def backward(self, grad_output):
        y = self._saved_tensor
        return grad_output * (1 - y**2)


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        self._saved_for_backward(input)
        return input @ self.W + self.b

    def backward(self, grad_output):
        input = self._saved_tensor
        self.grad_W = input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        return grad_output @ self.W.T

    def update(self, config):
        mm = config["momentum"]
        lr = config["learning_rate"]
        wd = config["weight_decay"]

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
