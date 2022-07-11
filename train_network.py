# -*- coding: utf-8 -*-
"""
@author: Charles Millard
"""
import os
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from Noisier2Noise.varnet_modified import VarNet

from Noisier2Noise.utils import *
from Noisier2Noise.zf_data_loader import zf_data

DTYPE = torch.float32


def end2end(config, trainloader, validloader, logdir):
    writer = SummaryWriter(logdir)

    with open(logdir + '/config', 'w') as fp:
        yaml.dump(config, fp, default_flow_style=False)

    base_net = VarNet(num_cascades=config['network']['ncascades'])
    network = pass_varnet
    base_net.to(config['network']['device'])

    # create optimizer
    optimizer = torch.optim.Adam(base_net.parameters(), lr=float(config['optimizer']['lr']),
                                 weight_decay=float(config['optimizer']['weight_decay']))
    criterion = mse_loss

    epoch_frac_save = 10
    nshow = trainloader.__len__() // epoch_frac_save if torch.cuda.is_available() else 1

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
            elif config['data']['method'] in ["ssdu", "ssdu_bern"]:
                loss = criterion(outputs * (y != 0), y)
            else:  # fully supervised
                loss = criterion(outputs, y0)

            running_loss += loss
            loss.backward()
            optimizer.step()

            if i % nshow == (nshow - 1):  # print every nshow mini-batches
                print('%d    |   %d    |%e ' % (epoch + 1, i + 1, running_loss / nshow))
                writer.add_scalar('Training_losses/MSE_loss', running_loss / nshow, epoch * epoch_frac_save + j)

                if not torch.cuda.is_available():  # truncate training when local
                    print('truncating training as gpu not available')
                    break

                running_loss = 0
                j += 1

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
                    y0_est = outputs * (y == 0) / (1 - K) + y
                elif config['data']['method'] in ["ssdu", "ssdu_bern"]:
                    outputs = network(y_tilde, base_net)
                    y0_est = outputs * (y == 0) + y
                else:
                    y0_est = network(y, base_net)

                y0_est[pad] = 0  # remove zero-padding found in y0
                running_loss_val += criterion(y0_est, y0)

                if not torch.cuda.is_available() and i % nshow == (nshow - 1):
                    break

        writer.add_scalar('Validation_losses/MSE_loss', running_loss_val / (i + 1), epoch)
        print('Validation loss is {:e}'.format(running_loss_val / (i + 1)))

        loss_val_all.append(running_loss_val)
        torch.save(base_net.state_dict(), logdir + '/state_dict')

        # save examples from validation set
        if epoch == 0:
            x0 = kspace_to_rss(y0[0:4])
            x0 = torchvision.utils.make_grid(x0).detach().cpu()
            xnorm = torch.max(x0)
            writer.add_image('ground_truth', x0 / xnorm)

            xzf = kspace_to_rss(y_tilde[0:4])
            xzf = torchvision.utils.make_grid(xzf).detach().cpu()
            writer.add_image('CNN input', xzf / xnorm)

        x0_est = kspace_to_rss(y0_est[0:4])
        x0_est = torchvision.utils.make_grid(x0_est).detach().cpu()
        writer.add_image('estimate', x0_est / xnorm)


if __name__ == '__main__':
    torch.set_default_dtype(DTYPE)
    torch.backends.cudnn.enabled = False

    config_name = 'config.yaml'
    config = load_config(config_name)
    print('using config file ', config_name)

    if torch.cuda.is_available():
        config['network']['device'] = 'cuda'
    else:
        config['network']['device'] = 'cpu'
        config['optimizer']['batch_size'] = 2  # small batch size so can debug locally

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    train_load = DataLoader(zf_data('train', config), batch_size=config['optimizer']['batch_size'], shuffle=True)
    valid_load = DataLoader(zf_data('val', config), batch_size=3 * config['optimizer']['batch_size'])

    logdir = "logs/" + config['network']['device'] + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    if ~os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)

    print('Saving results to directory ', logdir)
    end2end(config, train_load, valid_load, logdir)
