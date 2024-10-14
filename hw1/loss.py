from __future__ import division
import numpy as np


class KLDivLoss(object):
    def __init__(self, name):
        self.name = name
        self.eps = 1e-9

    def forward(self, input, target):
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        prob = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        self.prob = prob
        loss_batch = np.sum(
            target * (np.log(target + self.eps) - np.log(prob + self.eps)), axis=1
        )
        return np.mean(loss_batch)

    def backward(self, input, target):
        return (self.prob - target) / input.shape[0]


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self.eps = 1e-9

    def forward(self, input, target):
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        prob = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        self.prob = prob
        return -np.sum(target * np.log(prob + self.eps)) / input.shape[0]

    def backward(self, input, target):
        return (self.prob - target) / input.shape[0]


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        batch_size, num_classes = input.shape
        h_k = np.zeros_like(input)

        for i in range(batch_size):
            x_t_n = np.sum(input[i] * target[i])
            for j in range(num_classes):
                if 1 == target[i][j]:
                    continue
                h_k[i][j] = max(0, self.margin - x_t_n + input[i][j])

        self.h_k = h_k
        return np.mean(np.sum(h_k, axis=1))

    def backward(self, input, target):
        grad = np.where(self.h_k > 0, 1, 0)
        correct_grad = -np.sum(grad, axis=1, keepdims=True)
        grad = grad + correct_grad * target
        return grad / input.shape[0]


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        """Your codes here"""
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        """Your codes here"""
        pass
        # TODO END
