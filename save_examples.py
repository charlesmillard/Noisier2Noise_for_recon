import torch.cuda
import torchvision

from utils import *
from torch.utils.data import DataLoader
from fastmri.models.unet import Unet
from fastmri.models.varnet import VarNet
from zf_data_loader import zf_data

from skimage.metrics import structural_similarity as ssim

import sys
import os
import matplotlib

type = '8depthtest'
root = 'logs/cuda/'
if type == '4':
    root = 'logs/cuda/4.0x/'
    log_loc = [root + 'full/66595042_4.0', root + 'ssdu_bern/66595042_1.6',
               root + 'ssdu/65370413_1.6',
               root + 'n2n_unweighted/65370413_4.0',
               root + 'n2n_weighted/65370413_4.0']
elif type == '8':
    root = 'logs/cuda/8.0x/'
    log_loc = [root + 'full/66595042_4.0', root + 'ssdu_bern/66595042_2.0',
               root + 'ssdu/65370413_2.0',
               root + 'n2n_unweighted/65370413_10.0',
               root + 'n2n_weighted/65370413_10.0']
elif type == '8mistuned':
    root = 'logs/cuda/8.0x/'
    log_loc = [root + 'full/66595042_4.0', root + 'ssdu_bern/66595042_4.0',
               root + 'ssdu/65370413_4.0',
               root + 'n2n_unweighted/65370413_12.0',
               root + 'n2n_weighted/65370413_12.0']
elif type == '8depthtest':
    root = 'logs/cuda/'
    log_loc = [root + '8.0x/full/66595042_4.0',
               root + '73247503']

xz = 120
yz = 60
wth = 100

with torch.no_grad():
    for l in log_loc:
        config = load_config(l + '/config')
        config['network']['device'] = 'cpu'
        base_net = VarNet(num_cascades=config['network']['ncascades'])
        network = passVarnet

        sd = 300 # 260, 310, 320, 420
        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)

        test_load = DataLoader(zf_data('test', config), batch_size=1, shuffle=True)
        base_net.load_state_dict(torch.load(l + '/state_dict', map_location=config['network']['device']))
        base_net.to(config['network']['device'])
        base_net.eval()
        for i, data in enumerate(test_load, 0):
            y0, y, y_tilde, K = data
            y0 = y0.to(config['network']['device'])
            y = y.to(config['network']['device'])
            y_tilde = y_tilde.to(config['network']['device'])
            K = K.to(config['network']['device'])

            x0 = kspaceToRSS(y0)

            pad = torch.abs(y0) < torch.mean(torch.abs(y0)) / 100
            if config['data']['method'] == "n2n":
                outputs = network(y_tilde, base_net)
                y0_est = outputs * (y == 0) / (1 - K) + y
            elif config['data']['method'] in ["ssdu", "ssdu_bern"]:
                outputs = network(y_tilde, base_net)
                y0_est = outputs * (y == 0) + y
            else:
                y0_est = network(y, base_net)

            y0_est[pad] = 0  # set y0 padding to zero
            x0_est = kspaceToRSS(y0_est)

            mx = torch.max(torch.abs(x0[0]))
            if 'x0_est_all' in locals():
                x0_est_all = torch.cat((x0_est_all, x0_est/mx), dim=0)
                x0_est_zoom = torch.cat((x0_est_zoom, x0_est[:, :, xz:xz+wth, yz:yz+wth] / mx), dim=0)
            else:
                x0_est_all = torch.cat((x0/mx, x0_est/mx), dim=0)
                x0_est_zoom = torch.cat((x0[:, :, xz:xz + wth, yz:yz + wth]/mx, x0_est[:, :, xz:xz + wth, yz:yz + wth] / mx), dim=0)

            print('MSE is {:e}'.format(torch.mean((y0-y0_est)**2)))
            break

def saveIm(im, ndisp, name):
    im = torch.abs(im[0:ndisp])
    im = torchvision.utils.make_grid(im, nrow=4).detach().cpu()
    torchvision.utils.save_image(im, name)

print('seed is {}'.format(sd))
torchvision.utils.save_image(torch.as_tensor(x0_est_all), 'saved_images/x_' + str(sd) + '_' + type + '.png')
torchvision.utils.save_image(torch.as_tensor(x0_est_zoom), 'saved_images/x_' + str(sd) + '_' + type + '_zoom.png')