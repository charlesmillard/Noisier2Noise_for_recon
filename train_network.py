# -*- coding: utf-8 -*-
"""
Created on Nov 1st 2021

@author: Charles Millard
"""
import sys
import torchvision

from utils import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from fastmri.models.unet import Unet
from fastmri.models.varnet import VarNet
from zf_data_loader import zf_data

DTYPE = torch.float32


def end2end(config, trainloader, validloader, logdir):
    writer = SummaryWriter(logdir)

    with open(logdir + '/config', 'w') as fp:
        yaml.dump(config, fp, default_flow_style=False)

    multicoil = config['data']['multicoil']
    if config['network']['type'] == 'unet':
        chans = 2*config['data']['nc'] if multicoil else 2
        base_net = Unet(in_chans=chans, out_chans=chans)
        network = passUnet
    elif config['network']['type'] == 'varnet':
        base_net = VarNet(num_cascades=config['network']['ncascades'])
        network = passVarnet

    base_net.to(config['network']['device'])

    # create optimizer
    if config['optimizer']['name'] == 'Adam':
        optimizer = torch.optim.Adam(base_net.parameters(), lr=float(config['optimizer']['lr'])
                                     , weight_decay=float(config['optimizer']['weight_decay']))
    elif config['optimizer']['name'] == 'SGD':
        optimizer = torch.optim.SGD(base_net.parameters(), lr=float(config['optimizer']['lr'])
                                    , weight_decay=float(config['optimizer']['weight_decay']),
                                    momentum=float(config['optimizer']['momentum']))
    else:
        raise NameError('You have chosen an invalid optimizer name')

    if config['optimizer']['loss'] == 'mse':
        criterion = MSEloss
    elif config['optimizer']['loss'] == 'rmse':
        criterion = RMSEloss
    elif config['optimizer']['loss'] == 'l1':
        criterion = torch.nn.L1Loss()
    elif config['optimizer']['loss'] == 'l1l2':
        criterion = l12loss
    else:
        raise NameError('You have chosen an invalid loss')

    epoch_frac_save = 10
    if not torch.cuda.is_available():
        nshow = 1
        dev = 'cpu'
    else:
        nshow = trainloader.__len__() // epoch_frac_save
        dev = 'cuda'

    try:
        loc = 'logs/cuda/' + str(config['optimizer']['load_model_loc'])
        base_net.load_state_dict(torch.load(loc + '/state_dict_best', map_location=dev))
        print('model loading successful, using ' + str(config['optimizer'][
            'load_model_loc']) + ' as parameter initialisation')
    except:
        print('model loading unsucessful - using random parameter initialisation')

    loss_val_all = []
    for epoch in range(config['optimizer']['epochs']):
        print('training...')
        print('epoch|minibatch|Label loss')

        j = 0

        running_loss = 0

        base_net.train()
        for i, data in enumerate(trainloader, 0):
            y0, y, y_tilde, K = data
            y0 = y0.to(config['network']['device'])
            y = y.to(config['network']['device'])
            y_tilde = y_tilde.to(config['network']['device'])
            K = K.to(config['network']['device'])
            optimizer.zero_grad()

            outputs = network(y_tilde, base_net)
            if config['data']['method'] == "n2n":
                if config['optimizer']['weight_loss']:
                    loss = criterion(outputs / (1 - K), y / (1 - K))
                else:
                    loss = criterion(outputs, y)
            elif config['data']['method'] == "ssdu":
                loss = criterion(outputs * (y != 0), y)
            else:
                loss = criterion(outputs, y0)

            running_loss += loss

            loss.backward()
            optimizer.step()

            if i % nshow == (nshow - 1):  # print every nshow mini-batches
                print('%d    |   %d    |%e ' %
                      (epoch + 1, i + 1, running_loss / nshow))
                writer.add_scalar('Training_losses/MSE_loss', running_loss / nshow, epoch * epoch_frac_save + j)

                if not torch.cuda.is_available():  # truncate training when on my machine
                    if j == 0:
                        break

                running_loss = 0
                j += 1

        # validate undersampled to fully sampled map
        running_loss_val = 0
        with torch.no_grad():
            print('validation...')
            base_net.eval()
            for i, data in enumerate(validloader, 0):
                y0, y, y_tilde, K = data
                y0 = y0.to(config['network']['device'])
                y = y.to(config['network']['device'])
                y_tilde = y_tilde.to(config['network']['device'])
                K = K.to(config['network']['device'])

                pad = torch.abs(y0) < torch.mean(torch.abs(y0)) / 100
                if config['data']['method'] == "n2n":
                    outputs = network(y_tilde, base_net)
                    y0_est = outputs * (y == 0)/(1 - K) + y
                elif config['data']['method'] == "ssdu":
                    outputs = network(y_tilde, base_net)
                    y0_est = outputs * (y == 0) + y
                else:
                    y0_est = network(y, base_net)

                y0_est[pad] = 0
                running_loss_val += criterion(y0_est, y0)

                if not torch.cuda.is_available():
                    if i % nshow == (nshow - 1):
                        break

        writer.add_scalar('Validation_losses/MSE_loss', running_loss_val / (i + 1), epoch)
        print('Validation loss is {:e}'.format(running_loss_val / (i + 1)))

        loss_val_all.append(running_loss_val)

        torch.save(base_net.state_dict(), logdir + '/state_dict')
        if running_loss_val == min(loss_val_all):
            print('** Best validation performance so far **')
            torch.save(base_net.state_dict(), logdir + '/state_dict_best')

        # save examples from validation set
        if epoch == 0:
            x0 = kspaceToRSS(y0[0:4])
            x0 = torchvision.utils.make_grid(x0).detach().cpu()
            xnorm = torch.max(x0)
            writer.add_image('ground_truth', x0 / xnorm)

            xzf = kspaceToRSS(y_tilde[0:4])
            xzf = torchvision.utils.make_grid(xzf).detach().cpu()
            writer.add_image('CNN input', xzf / xnorm)

        x0_est = kspaceToRSS(y0_est[0:4])
        x0_est = torchvision.utils.make_grid(x0_est).detach().cpu()
        writer.add_image('estimate', x0_est / xnorm)

    return network


if __name__ == '__main__':
    np.random.seed(463)
    torch.set_default_dtype(DTYPE)
    torch.backends.cudnn.enabled = False

    config_name = 'config.yaml'

    config = load_config(config_name)
    print('using config file ', config_name)

    if len(sys.argv) >= 3:
        config['data']['us_fac_lambda'] = float(sys.argv[2])/10
        print('Changed undersampling of lambda to {}'.format(sys.argv[2]))

    if torch.cuda.is_available():
        config['network']['device'] = 'cuda'
    else:
        config['network']['device'] = 'cpu'
        config['optimizer']['batch_size'] = 2  # small batch size so can handle on my machine

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    train_load = DataLoader(zf_data('train', config), batch_size=config['optimizer']['batch_size'], shuffle=True)
    valid_load = DataLoader(zf_data('val', config), batch_size=3*config['optimizer']['batch_size'])

    logdir = "logs/" + config['network']['device'] + '/'
    if len(sys.argv) == 1:
        logdir = logdir + datetime.now().strftime("%Y%m%d-%H%M%S")
    elif len(sys.argv) == 2:
        logdir = logdir + str(sys.argv[1])
    else:
        logdir = logdir + str(sys.argv[1]) + '_lambda_us_' + str(sys.argv[2])

    print('Saving results to directory ', logdir)

    network = end2end(config, train_load, valid_load, logdir)
