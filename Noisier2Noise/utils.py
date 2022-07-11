import yaml
import numpy as np
import torch
import torch.nn.functional as F


def load_config(cname):
    """ loads yaml file """
    with open(cname, 'r') as stream:
        configs = yaml.safe_load_all(stream)
        config = next(configs)
    return config


def pad_or_trim_tensor(x, nx_target, ny_target):
    _, _, nx, ny = x.shape
    if nx != nx_target or ny != ny_target:
        x = kspace_to_im(torch.unsqueeze(x, 0))
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

        x = im_to_kspace(x)
    return torch.squeeze(x)


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


def kspace_to_im(y):
    y = torch.permute(y, (0, 2, 3, 4, 1)).contiguous()
    y = torch.view_as_complex(y)
    x = torch.view_as_real(torch.fft.fftshift(torch.fft.ifft2(y, norm='ortho')))
    x = torch.permute(x, (0, 4, 1, 2, 3))
    return x


def im_to_kspace(x):
    x = torch.permute(x, (0, 2, 3, 4, 1)).contiguous()
    x = torch.view_as_complex(x)
    y = torch.view_as_real(torch.fft.fft2(torch.fft.ifftshift(x), norm='ortho'))
    y = torch.permute(y, (0, 4, 1, 2, 3))
    return y


def kspace_to_rss(y):
    x = kspace_to_im(y)
    x = torch.permute(x, (0, 2, 3, 4, 1)).contiguous()
    x = torch.view_as_complex(x)
    x = torch.sqrt(torch.sum(torch.abs(x) ** 2, 1))
    (_, nx, ny) = x.shape
    x = x[:, nx//2 - 160:nx//2+160, ny//2-160:ny//2+160]
    return x.unsqueeze(1)


def pass_varnet(y, base_net):
    y_tild = torch.permute(y, (0, 2, 3, 4, 1))
    outputs = base_net(y_tild, (y_tild != 0).bool())
    outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
    return outputs * (y == 0) + y


def nmse(x, x0):
    return np.mean(np.abs(x - x0)**2)/np.mean(np.abs(x0)**2)


def mse_loss(xhat, x0):
    return torch.mean(torch.abs(xhat-x0)**2)
