import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_ch=1280, resolution=1, nonlinearity="relu"):
        super(MLP, self).__init__()
        output_dim=output_ch*resolution*resolution
        self.resolution=resolution
        self.output_ch=output_ch
        inter_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, inter_dim, bias=False)
        # self.bn = nn.BatchNorm1d(inter_dim)
        self.fc2 = nn.Linear(inter_dim, output_dim, bias=False)

    def forward(self, x, x_ts):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        # x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)

class MLP_one_hot(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=1280, output_ch=1280, resolution=1, nonlinearity="relu"):
        super(MLP_one_hot, self).__init__()
        output_dim=output_ch*resolution*resolution
        self.resolution=resolution
        self.output_ch=output_ch
        inter_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, x_ts):
        x = x.to(self.fc1.weight.dtype)
        x = self.fc1(x)
        return x.view(x.shape[0], self.output_ch, self.resolution, self.resolution)

model_types={
    "MLP":MLP,
    "MLP_one_hot":MLP_one_hot
}