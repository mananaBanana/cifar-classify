import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        # Initialize parameters
        self.imshape = params["shape"]
        self.num_channels = self.imshape[2]
        self.out_channels = params["out_channels"]
        self.pool_size = params["pool_size"]
        self.class_num = params["class_num"]
        self.kern_size = params["kern_size"]
        self.layer_shape = [0, 0]  # Shape after convulution
        self.batch_norm = params["batch_norm"]

        # Convulutional layers
        self.conv1 = nn.Conv2d(self.num_channels, self.out_channels, self.kern_size)

        self.layer_shape[0] = getsizeafterconv(self.imshape[0], self.kern_size, self.pool_size)
        self.layer_shape[1] = getsizeafterconv(self.imshape[1], self.kern_size, self.pool_size)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels * 2, self.kern_size)
        self.layer_shape[0] = getsizeafterconv(self.layer_shape[0], self.kern_size, self.pool_size)
        self.layer_shape[1] = getsizeafterconv(self.layer_shape[1], self.kern_size, self.pool_size)

        self.fc1 = nn.Linear(self.layer_shape[0] * self.layer_shape[1] * self.out_channels * 2, self.out_channels * 4)
        self.fc2 = nn.Linear(self.out_channels * 4, self.out_channels * 3)
        self.fc3 = nn.Linear(self.out_channels * 3, self.class_num)


    def forward(self, s):
        s = F.max_pool2d(self.conv1(s), self.pool_size)
        s = F.relu(s)
        s = F.max_pool2d(self.conv2(s), self.pool_size)
        s = F.relu(s)

        s = s.view(-1, self.layer_shape[0] * self.layer_shape[1] * self.out_channels * 2)
        s = self.fc1(s)
        s = F.relu(s)

        s = F.relu(self.fc2(s))

        return self.fc3(s)


# Supporting functions
def getsizeafterconv(imsize, filtsize, poolsize, filtstride=1, poolstride=2):
    """ Get the input channel size after convolution and pooling"""
    return int((((imsize - filtsize) / filtstride + 1) - poolsize) / poolstride + 1)
