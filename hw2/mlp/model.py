import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Parameters
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        # Store the average mean and variance
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # Initialize your parameter
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
        if self.training:
            mean = input.mean(dim=0)
            var = input.var(dim=0)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

            input = (input - mean) / (var + self.eps).sqrt()
        else:
            input = (input - self.running_mean) / (self.running_var + self.eps).sqrt()

        return input * self.weight + self.bias


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
        if not self.training:
            return input
        mask = torch.bernoulli(
            torch.full_like(input, 1 - self.p, dtype=torch.float32)
        ) / (1 - self.p)
        mask = mask.to(input.device)
        return input * mask


class Model(nn.Module):
    def __init__(self, drop_rate=0.5):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.bn = BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.dropout = Dropout(drop_rate)
        self.fc2 = nn.Linear(1024, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # the 10-class prediction output is named as "logits"
        logits = self.fc2(self.dropout(self.relu(self.bn(self.fc1(x)))))

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y.long())
        correct_pred = pred.int() == y.int()
        acc = torch.mean(
            correct_pred.float()
        )  # Calculate the accuracy in this mini-batch

        return loss, acc
