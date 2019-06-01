import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd.variable import Variable
import numpy as np

from models.data_loader import CIFARLoader
from models.model import Net


def train(model, datasetloader, loss_fn, optimizer, batch_size=4, num_epochs=3):
    """ Train on the dataset for num_epochs printing the loss for every epoch"""

    dataloader = datasetloader.get_dataloader(batch_size)['train']  # load train_set

    # set model in training mode
    model.train()
    running_loss = 0.0
    loss = 0
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(iter(dataloader)):
            optimizer.zero_grad()  # Reset gradients in every iteration
            train_batch, train_labels = Variable(images), Variable(labels)

            output_batch = model(train_batch)
            loss = loss_fn(output_batch, train_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 1000 == 999:
                print("Epoch:{} Batch:{} loss: {}".format(epoch + 1, batch_idx + 1, running_loss / 2000))
                running_loss = 0.0

        print("Epoch {} done with loss {}".format(epoch + 1, loss))


def evaluate(model, datasetloader, loss_fn, batch_size=4):
    dataloader = datasetloader.get_dataloader(batch_size)['test']
    model.eval()

    batch_accuracy = []

    running_loss = 0.0
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)
            running_loss += loss.item()
            batch_accuracy.append(accuracy(output_batch, labels_batch))
        mean_accuracy = np.mean(batch_accuracy)
        print("Mean accuracy {}".format(mean_accuracy))


def accuracy(outputs, labels):
    _, pred = torch.max(outputs.data, dim=1)
    acc_sum = torch.sum(pred == labels)
    labels_len = float(len(labels))
    return acc_sum.item() / labels_len


if __name__ == "__main__":
    params = {"shape": (32, 32, 3), "out_channels": 10, "pool_size": 2, "class_num": 10, "kern_size": 5,
              "batch_norm": True}

    net = Net(params)

    # Specify transformations by the image
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ]

    dataloader = CIFARLoader(transform_list=transform_list)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    train(net, dataloader, loss_fn, optimizer, 16, 10)
    evaluate(net, dataloader, loss_fn, 16)
