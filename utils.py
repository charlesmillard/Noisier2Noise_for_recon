# import torch
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as nd

def load_config(cname):
    """ loads yaml file """
    with open(cname, 'r') as stream:
        configs = yaml.safe_load_all(stream)
        config = next(configs)

    return config

def RMSEloss(xhat, x0):
    return torch.sqrt(torch.mean(torch.abs(xhat-x0)**2))

def MSEloss(xhat, x0):
    return torch.mean(torch.abs(xhat-x0)**2)

def l12loss(xhat, x0):
    l2 = torch.linalg.vector_norm((xhat - x0), 2)/torch.linalg.vector_norm(x0, 2)
    l1 = torch.linalg.vector_norm((xhat - x0), 1) / torch.linalg.vector_norm(x0, 1)
    return l1 + l2

def pad_or_trim(x, nx_target, ny_target):
    nx, ny = x.shape

    dx = nx_target - nx
    if dx > 0:
        x = np.pad(x, ((dx//2, dx//2 + dx % 2), (0, 0)))
    elif dx < -1:
        x = x[-dx//2:(dx+2)//2 - 1]
    elif dx == -1:
        x = x[:-1]

    dy = ny_target - ny
    if dy > 0:
        x = np.pad(x, ((0, 0), (dy//2, dy//2 + dy % 2)))
    elif dy < -1:
        x = x[:, -dy//2:(dy+2)//2 - 1]
    elif dy == -1:
        x = x[:, :-1]

    return x

def pad_or_trim_tensor(x, nx_target, ny_target):
    _, _, nx, ny = x.shape

    if nx != nx_target or ny != ny_target:
        x = kspaceToIm(torch.unsqueeze(x, 0))

        dx = nx_target - nx
        if dx > 0:
            x = F.pad(x, (0, 0, dx//2, dx//2 + dx % 2))
        elif dx < -1:
            x = x[:, :, :, -dx//2:(dx+2)//2 - 1]
        elif dx == -1:
            x = x[:, :, :-1]

        dy = ny_target - ny
        if dy > 0:
            x = F.pad(x, (dy//2, dy//2 + dy % 2))
        elif dy < -1:
            x = x[:, :, :, :, -dy//2:(dy+2)//2 - 1]
        elif dy == -1:
            x = x[:, :, -1]

        x = imToKspace(x)

    return torch.squeeze(x)


def remove_padding(y):
    (_, _, nx, ny) = y.shape
    left = torch.sum(torch.cumsum(y[0, 0, 0], dim=0) == 0)
    right = torch.sum(torch.cumsum(torch.fliplr(y)[0, 0, 0], dim=0) == 0)
    y = y[:, :, :,  left:-right]
    # print(left, right)
    return y


def fftnc(x):
    # normalised, centered fft
    x = np.fft.fftshift(x, axes=(0, 1,))
    k = np.fft.fft2(x, norm="ortho", axes=(0, 1,))
    k = np.fft.fftshift(k, axes=(0, 1,))
    return k


def ifftnc(k):
    # normalised, centered ifft
    k = np.fft.ifftshift(k, axes=(0, 1,))
    x = np.fft.ifft2(k,  norm="ortho", axes=(0, 1,))
    x = np.fft.ifftshift(x, axes=(0, 1,))
    return x

def kspaceToIm(y):
    multicoil = (len(y.shape) == 5)
    y = torch.permute(y, (0, 2, 3, 4, 1)).contiguous() if multicoil else torch.permute(y, (0, 2, 3, 1)).contiguous()
    y = torch.view_as_complex(y)
    x = torch.view_as_real(torch.fft.fftshift(torch.fft.ifft2(y, norm='ortho')))
    x = torch.permute(x, (0, 4, 1, 2, 3)) if multicoil else torch.permute(x, (0, 3, 1, 2))
    return x

def imToKspace(x):
    multicoil = (len(x.shape) == 5)
    x = torch.permute(x, (0, 2, 3, 4, 1)).contiguous() if multicoil else torch.permute(x, (0, 2, 3, 1)).contiguous()
    x = torch.view_as_complex(x)
    y = torch.view_as_real(torch.fft.fft2(torch.fft.ifftshift(x), norm='ortho'))
    y = torch.permute(y, (0, 4, 1, 2, 3)) if multicoil else torch.permute(y, (0, 3, 1, 2))
    return y

def passUnet(y, base_net):
    multicoil = (len(y.shape) == 5)
    if multicoil:
        x = kspaceToIm(y)
        (n1, n2, n3, n4, n5) = x.shape
        x = torch.reshape(x, (n1, n2 * n3, n4, n5))
        outputs = base_net(x)
        outputs = torch.reshape(outputs, (n1, n2, n3, n4, n5))
        outputs = imToKspace(outputs)
    else:
        outputs = imToKspace(base_net(kspaceToIm(y)))
    return outputs * (y == 0) + y

def passVarnet(y, base_net):
    multicoil = (len(y.shape) == 5)
    if multicoil:
        y_tild = torch.permute(y, (0, 2, 3, 4, 1))
        outputs = base_net(y_tild, (y_tild != 0).bool())
        outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
    else:
        y_tild = torch.unsqueeze(y, 0)
        y_tild = torch.permute(y_tild, [1, 0, 3, 4, 2])
        outputs = base_net(y_tild, (y_tild != 0).bool())
        outputs = torch.squeeze(outputs)
        outputs = torch.permute(outputs, [0, 3, 1, 2])
    return outputs * (y == 0) + y

def kspaceToRSS(y):
    x = kspaceToIm(y)
    multicoil = (len(x.shape) == 5)
    x = torch.permute(x, (0, 2, 3, 4, 1)).contiguous() if multicoil else torch.permute(x, (0, 2, 3, 1)).contiguous()

    x = torch.view_as_complex(x)
    x = torch.sqrt(torch.sum(torch.abs(x) ** 2, 1)) if multicoil else torch.sqrt(torch.abs(x) ** 2)
    (_, nx, ny) = x.shape
    x = x[:, nx//2 - 160:nx//2+160, ny//2-160:ny//2+160]
    return x.unsqueeze(1)

def hfen(x, x0):
    loG = nd.gaussian_laplace(x - x0, sigma=1.5)
    loss = np.sqrt(np.mean(loG**2)/np.mean(x0**2))
    return loss

def nmse(x, x0):
    return np.mean(np.abs(x - x0)**2)/np.mean(np.abs(x0)**2)
