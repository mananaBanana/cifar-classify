import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd.variable import Variable
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from models.model import Net
from models.data_loader import CIFARLoader

def train(model, datasetloader, loss_fn, optimizer, batch_size=4, num_epochs=3):
    """ Train on the dataset for num_epochs printing the loss for every epoch"""

    dataloader = datasetloader.get_dataloader(batch_size)['train'] # load train_set

    # set model in training mode
    model.train()
    running_loss = 0.0
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(iter(dataloader)):
            optimizer.zero_grad() # Reset gradients in every iteration
            train_batch, train_labels = Variable(images), Variable(labels) 

            output_batch = model(train_batch)
            loss = loss_fn(output_batch, train_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if(batch_idx % 1000 == 999):
                print("Epoch:{} Batch:{} loss: {}".format(epoch + 1, batch_idx + 1, running_loss / 2000))
                running_loss = 0.0

        print("Epoch {} done with loss {}".format(epoch, loss))


def evaluate(model, datasetloader, loss_fn, batch_size=4):
    
    dataloader = datasetloader.get_dataloader(batch_size)['test']
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for data_batch, labels_batch in dataloader:

            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            
            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)
            running_loss += loss.item()
            print("Accuracy: {}".format(accuracy(output_batch, labels_batch)))

def accuracy(outputs, labels):
    _, pred = torch.max(outputs.data, dim=1)
    return torch.sum(output == labels) / labels.size

if __name__ == '__main__':
    params = {}
    params["shape"] = (32, 32, 3) 
    params["out_channels"] = 10
    params["pool_size"] = 2
    params["class_num"] = 10
    params["kern_size"] = 5
    params["batch_norm"] = True

    net = Net(params)

    # Specify transformations by the image
    transform_list = [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5) , (0.5, 0.5, 0.5))
            ]

    dataloader = CIFARLoader(transform_list=transform_list)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    train(net, dataloader, loss_fn, optimizer, 16, 5)
    evaluate(net, dataloader, loss_fn, 16)
