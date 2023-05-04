# -*- coding: utf-8 -*-
"""
@author: Charles Millard
"""
import os
import sys

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from Noisier2Noise.varnet_complex_output import VarNet
from Noisier2Noise.utils import *
from Noisier2Noise.zf_data_loader import subSampledData
from Noisier2Noise.noisier2noise_losses import *

DTYPE = torch.float32


def train_net(config, logdir):
    writer = SummaryWriter(logdir)

    with open(logdir + '/config', 'w') as fp:
        yaml.dump(config, fp, default_flow_style=False)

    network = VarNet(num_cascades=config['network']['ncascades'])
    network.to(dev)

    # create optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=float(config['optimizer']['lr']),
                                 weight_decay=float(config['optimizer']['weight_decay']))
    criterion = mse_loss

    # data loaders
    # print('len(subSampledData(train, config)):',len(subSampledData('train', config))) #0
    trainloader = DataLoader(dataset=subSampledData('train', config), batch_size=config['optimizer']['batch_size'], shuffle=True)
    validloader = DataLoader(dataset=subSampledData('val', config), batch_size=config['optimizer']['batch_size'])

    epoch_frac_save = 10 # save progress to writer every 10% through epoch
    nshow = trainloader.__len__() // epoch_frac_save if torch.cuda.is_available() else 1

    set_seeds(config['seed'])

    for epoch in range(config['optimizer']['epochs']):
        print('training...')
        print('epoch|minibatch|Label loss')

        j = 0
        running_loss = 0
        network.train()
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            loss = training_loss(data, network, config, criterion)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % nshow == (nshow - 1):  # print every nshow mini-batches
                print('%d    |   %d    |%e ' % (epoch + 1, i + 1, running_loss / nshow))
                writer.add_scalar('Training_losses/MSE_loss', running_loss / nshow, epoch * epoch_frac_save + j)
                if not torch.cuda.is_available():  # truncate training when local (for debugging)
                    print('truncating training as GPU not available')
                    break
                running_loss = 0
                j += 1

        with torch.no_grad():
            print('validation...')
            running_loss_val = 0
            running_loss_val_hat = 0
            network.eval()
            for i, data in enumerate(validloader, 0):
                loss_val, loss_val_hat, examples = val_loss(data,  network, config, criterion)

                running_loss_val += loss_val
                running_loss_val_hat += loss_val_hat

                if not torch.cuda.is_available() and i % nshow == (nshow - 1):
                    break

        # save progress
        writer.add_scalar('Validation_losses/MSE_loss', running_loss_val / (i + 1), epoch)
        writer.add_scalar('Validation_losses/MSE_loss_hat', running_loss_val_hat / (i + 1), epoch)
        print('Validation loss with y_tilde input is {:e}'.format(running_loss_val / (i + 1)))
        print('Validation loss with y input is {:e}'.format(running_loss_val_hat / (i + 1)))

        # save network parameters
        torch.save(network.state_dict(), logdir + '/state_dict')

        # save example recons
        save_examples(writer, examples)

        print('epoch ' + str(epoch) + ' complete')

    return 0


if __name__ == '__main__':
    torch.set_default_dtype(DTYPE)
    torch.backends.cudnn.enabled = False

    # load config file
    # config_name = sys.argv[1]
    config_name = '1D_partitioned_ssdu'
    config = load_config('configs/' + config_name + '.yaml')
    print('using config file ', config_name)

    # create log directory
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # logdir = "logs/" + dev + '/' + config_name + '/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = config['save']['logdir'] # can change save name liu 5.3
    if ~os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)
    print('Saving results to directory ', logdir)

    # train network
    train_net(config, logdir)

    print('Training complete')
