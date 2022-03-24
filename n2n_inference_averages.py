import torch.cuda
import torchvision

from utils import *
from torch.utils.data import DataLoader
from fastmri.models.unet import Unet
from fastmri.models.varnet import VarNet
from zf_data_loader import zf_data

from skimage.metrics import structural_similarity as ssim

import sys
import matplotlib.pyplot as plt
import os
import matplotlib

if len(sys.argv) == 1:
    log_loc = 'logs/cuda/significant_logs_columns_multicoil_6casc/ssdu_multilambda_4/60008780_lambda_us_2'
else:
    if torch.cuda.is_available():
        log_loc = "logs/cuda/" + str(sys.argv[1])
    else:
        log_loc = "logs/cpu/" + str(sys.argv[1])

config = load_config(log_loc + '/config')

if 'method' not in config['data']:
    config['data']['method'] = 'n2n'

if 'type' not in config['network']:
    config['network']['type'] = 'unet'

if 'multicoil' not in config['data']:
    config['data']['multicoil'] = False

if 'nx' not in config['data']:
    config['data']['nx'] = 388

if 'ny' not in config['data']:
    config['data']['ny'] = 640

if 'fully_samp_size' not in config['data']:
    config['data']['fully_samp_size'] = 12

if 'poly_order' not in config['data']:
    config['data']['poly_order'] = 8

if 'ncascades' not in config['network']:
    config['network']['ncascades'] = 4

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

if torch.cuda.is_available():
    config['network']['device'] = 'cuda'
else:
    config['network']['device'] = 'cpu'
    config['optimizer']['batch_size'] = 1

loss = 0
loss_sing = 0
loss_sing_rss = 0
all_ssim = []
all_ssim_sing = []
with torch.no_grad():
    multicoil = config['data']['multicoil']
    if config['network']['type'] == 'unet':
        chans = 2 * config['data']['nc'] if multicoil else 2
        base_net = Unet(in_chans=chans, out_chans=chans)
        network = passUnet
    elif config['network']['type'] == 'varnet':
        base_net = VarNet(num_cascades=config['network']['ncascades'])
        print(config['network']['ncascades'])
        network = passVarnet

    sd = 200 #110, 140, 200
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

    base_net.load_state_dict(torch.load(log_loc + '/state_dict', map_location=config['network']['device']))
    base_net.to(config['network']['device'])

    # number of parameters in model
    pp = 0
    for p in list(base_net.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    print('{} parameters in model'.format(pp))

    test_load = DataLoader(zf_data('test', config), batch_size=1, shuffle=False)

    loss_track = []
    nrep = 100
    y0_est_sum = 0
    t = 0
    for r in range(nrep):
        for i, data in enumerate(test_load, 0):
            if i == 2:
                y0, y, y_tilde, K = data
                y0, y, y_tilde, K = data
                y0 = y0.to(config['network']['device'])
                y = y.to(config['network']['device'])
                y_tilde = y_tilde.to(config['network']['device'])
                K = K.to(config['network']['device'])

                x0 = kspaceToRSS(y0)

                pad = torch.abs(y0) < torch.mean(torch.abs(y0)) / 100

                outputs = network(y_tilde, base_net)
                outputs[pad] = 0
                y0_est = (outputs - K * y_tilde) / (1 - K)
                y0_est = y0_est * (y == 0) + y
                y0_est[pad] = 0 # set y0 padding to zero
                y0_est_sum = y0_est_sum + y0_est
                t += 1
                loss_track.append(criterion(y0_est_sum/t, y0))
                print(loss_track)

x0_est = kspaceToRSS(y0_est)
plt.figure()
plt.plot(loss_track)
plt.figure()
plt.imshow(x0_est[0,0])
plt.show()