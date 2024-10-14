from network import Network
from utils import LOG_INFO
from layers import Selu, HardSwish, Linear, Tanh
from loss import KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

train_data, test_data, train_label, test_label = load_mnist_2d("data")

# Your model defintion here
model = Network()
model.add(Linear("fc1", 784, 128, 0.01))
model.add(Tanh("tanh1"))
model.add(Linear("fc2", 128, 10, 0.01))

loss = KLDivLoss(name="loss")

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "batch_size": 64,
    "max_epoch": 30,
    "disp_freq": 100,
    "test_epoch": 1,
}

for epoch in range(config["max_epoch"]):
    LOG_INFO("Training @ %d epoch..." % (epoch))
    train_net(
        model,
        loss,
        config,
        train_data,
        train_label,
        config["batch_size"],
        config["disp_freq"],
    )

    if epoch % config["test_epoch"] == 0:
        LOG_INFO("Testing @ %d epoch..." % (epoch))
        test_net(model, loss, test_data, test_label, config["batch_size"])
