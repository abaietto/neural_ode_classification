"""
ResNet and ODENet classes for ECG classification.

Code adapted from:
https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint


def norm(dim):
    """
    Group normalization to improve model accuracy and training speed.
    """
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    """
    Simple residual block used to construct ResNet.
    """
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.gn1 = norm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.gn2 = norm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Shortcut
        identity = x

        # First convolution
        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        # Second convolution
        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # Add shortcut
        out += identity

        return out


class ConcatConv1d(nn.Module):
    """
    1d convolution concatenated with time for usage in ODENet.
    """
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=True, transpose=False):
        super(ConcatConv1d, self).__init__()
        module = nn.ConvTranspose1d if transpose else nn.Conv1d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):
    """
    Network architecture for ODENet.
    """
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv1d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv1d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0    # Number of function evaluations

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODENet(nn.Module):
    """
    Neural ODE.

    Uses ODE solver (dopri5 by default) to yield model output.
    Backpropagation is done with the adjoint method as described in
    https://arxiv.org/abs/1806.07366.

    Parameters
    ----------
    odefunc : nn.Module
        network architecture
    rtol : float
        relative tolerance of ODE solver
    atol : float
        absolute tolerance of ODE solver
    """
    def __init__(self, odefunc, rtol=1e-3, atol=1e-3):
        super(ODENet, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint_adjoint(self.odefunc, x, self.integration_time, self.rtol, self.atol)
        return out[1]

    # Update number of function evaluations (nfe)
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):
    """
    Flatten feature maps for input to linear layer.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

    
def count_parameters(model):
    """
    Count number of tunable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

