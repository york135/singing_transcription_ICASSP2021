import torch.nn as nn
import torch
import torch.nn.functional as F

class MyDNN(nn.Module):
    def __init__(self):
        super(MyDNN, self).__init__()
        # Well, the three conv layers are actually useless......
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))
        self.fc1   = nn.Linear(11*60, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 128)
        self.fc4   = nn.Linear(128, 1)

    def forward(self, x):
        #out= F.relu(self.first(x))
        #out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv1(x))
        # out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv3(out))
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary
    model = MyDNN().cuda()
    # summary(model, input_size=(1, 512, 11))
