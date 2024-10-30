import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(BatchNorm2d, self).__init__()
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
        # input: [batch_size, num_feature_map, height, width]
        if self.training:
            mean = input.mean(dim=(0, 2, 3), keepdim=True)
            var = input.var(dim=(0, 2, 3), keepdim=True)

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
            )

            input = (input - mean) / (var + self.eps).sqrt()
        else:
            input = (input - self.running_mean.view(1, -1, 1, 1)) / (
                self.running_var.view(1, -1, 1, 1) + self.eps
            ).sqrt()

        return input * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        # input: [batch_size, num_feature_map, height, width]
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
        # input: [BS, 3, 32, 32]
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)  # out: [BS, 64, 32, 32]
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = Dropout(drop_rate)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # out: [BS, 64, 16, 16]

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)  # out: [BS, 128, 16, 16]
        self.bn2 = BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = Dropout(drop_rate)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # out: [BS, 128, 8, 8]

        self.fc = nn.Linear(128 * 8 * 8, 10)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # the 10-class prediction output is named as "logits"
        x = self.pool1(self.dropout1(self.relu1(self.bn1(self.conv1(x)))))  # 1st conv
        x = self.pool2(self.dropout2(self.relu2(self.bn2(self.conv2(x)))))  # 2nd conv
        x = x.view(x.size(0), -1)  # flatten
        logits = self.fc(x)

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y.long())
        correct_pred = pred.int() == y.int()
        acc = torch.mean(
            correct_pred.float()
        )  # Calculate the accuracy in this mini-batch

        return loss, acc
