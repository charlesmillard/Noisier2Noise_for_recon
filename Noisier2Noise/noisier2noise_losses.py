import torch
from Noisier2Noise.utils import pass_varnet


def training_loss(data, network, config, criterion):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    y0, y, y_tilde, K = data
    y0 = y0.to(dev)
    y = y.to(dev)
    y_tilde = y_tilde.to(dev)
    K = K.to(dev)

    pad_mask = (y0 != 0)

    if config['data']['method'] == "n2n":
        outputs = pass_varnet(y_tilde, network)
        outputs *= pad_mask
        if config['optimizer']['weight_loss']:
            loss = criterion(outputs / (1 - K), y / (1 - K))
        else:
            loss = criterion(outputs, y)
    elif config['data']['method'] == "ssdu":
        outputs = pass_varnet(y_tilde, network)
        outputs *= pad_mask
        ssdu_mask = (y != 0) * (y_tilde == 0)
        loss = criterion(ssdu_mask * outputs, ssdu_mask * y)
    elif config['data']['method'] == "full":  # fully supervised
        outputs = network(y, network)
        outputs *= pad_mask
        loss = criterion(outputs, y0)
    else:
        raise ValueError('The method ' + config['data']['method'] + 'is invalid, must be one of {n2n, ssdu, full}')

    return loss


def val_loss(data, network, config, criterion):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    y0, y, y_tilde, K = data
    y0 = y0.to(dev)
    y = y.to(dev)
    y_tilde = y_tilde.to(dev)
    K = K.to(dev)

    pad_mask = (y0 != 0)

    if config['data']['method'] == "n2n":
        outputs = pass_varnet(y_tilde, network)
        ydc = outputs * (y == 0) / (1 - K) + y  # estimate with y_tilde input

        outputs_hat = pass_varnet(y, network)
        yhat = (outputs_hat - K * y) / (1 - K)  # estimate with y input
    elif config['data']['method'] == "ssdu":
        outputs = pass_varnet(y_tilde, network)
        ydc = outputs * (y == 0) + y

        outputs_hat = pass_varnet(y, network)
        yhat = outputs_hat
    elif config['data']['method'] == "full":
        yhat = pass_varnet(y, network)
        ydc = yhat
    else:
        raise ValueError('The method ' + config['data']['method'] + 'is invalid, must be one of {n2n, ssdu, full}')

    ydc *= pad_mask
    yhat *= pad_mask

    return criterion(ydc, y0), criterion(yhat, y0), [y0, y_tilde, ydc, yhat]
