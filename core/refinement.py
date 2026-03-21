import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinementHead(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=32):
        super(RefinementHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, image1, disp):
        x = torch.cat((image1, disp), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        res = self.conv5(x)
        return disp + res
