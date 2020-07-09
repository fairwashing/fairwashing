import torch.nn.functional as F
import torch.nn as nn
import math


class VGG16(nn.Module):
    def __init__(self, activation_fn='relu', beta=20):
        super(VGG16, self).__init__()
        activation_layer = None
        if activation_fn == 'relu':
            activation_layer = nn.ReLU()
        elif activation_fn == 'softplus':
            activation_layer = nn.Softplus(beta=beta)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            activation_layer,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation_layer,

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_layer,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            activation_layer,

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_layer,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_layer,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_layer,

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            activation_layer,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_layer,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_layer,

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_layer,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_layer,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_layer,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            activation_layer,
            nn.Dropout(),
            nn.Linear(512, 512),
            activation_layer,
            nn.Linear(512, 10),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, activation_fn='relu', beta=20):
        super(ConvNet, self).__init__()

        if activation_fn == 'relu':
            activation_layer = nn.ReLU()
        elif activation_fn == 'softplus':
            activation_layer = nn.Softplus(beta=beta)

        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            activation_layer,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            activation_layer,
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4*4*50, 500),
            activation_layer,
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
