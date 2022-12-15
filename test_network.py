"""
@author: Charles Millard
"""

import sys

from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from Noisier2Noise.varnet_complex_output import VarNet
from Noisier2Noise.utils import *
from Noisier2Noise.zf_data_loader import subSampledData
from Noisier2Noise.noisier2noise_losses import val_loss


def test_net(config, log_loc):
    criterion = mse_loss

    sd = 380
    set_seeds(sd)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        network = VarNet(num_cascades=config['network']['ncascades'])

        network.load_state_dict(torch.load(log_loc + '/state_dict', map_location=dev))
        network.to(dev)
        network.eval()

        testloader = DataLoader(subSampledData('test', config), batch_size=config['optimizer']['batch_size'],
                                shuffle=False)

        loss_dc = []
        loss_hat = []
        all_ssim_dc = []
        all_ssim_hat = []
        for i, data in enumerate(testloader, 0):
            print('testing slice ' + str(i))
            loss_val, loss_val_hat, examples = val_loss(data, network, config, criterion)
            y0, y_tilde, ydc, yhat = examples

            x0 = kspace_to_rss(y0)
            x_dc = kspace_to_rss(ydc)
            x_hat = kspace_to_rss(yhat)
            for ii in range(x0.shape[0]):
                x2 = np.array((x0[ii, 0]).detach().cpu())

                x1 = np.array((x_dc[ii, 0]).detach().cpu())
                all_ssim_dc.append(ssim(x1, x2, data_range=x2.max()))
                loss_dc.append(nmse(np.asarray(ydc[ii].detach().cpu()), np.asarray(y0[ii].detach().cpu())))

                x1 = np.array((x_hat[ii, 0]).detach().cpu())
                all_ssim_hat.append(ssim(x1, x2, data_range=x2.max()))
                loss_hat.append(nmse(np.asarray(yhat[ii].detach().cpu()), np.asarray(y0[ii].detach().cpu())))

    np.savez(log_loc + '/results.npz', loss_dc=loss_dc, loss_hat=loss_hat,
             all_ssim_dc=all_ssim_dc, all_ssim_hat=all_ssim_hat)

    print('Results saved in ' + log_loc)


if __name__ == '__main__':
    log_loc = sys.argv[1]
    print('Testing network saved in ' + log_loc)

    config = load_config(log_loc + '/config')
    if not torch.cuda.is_available():
        config['optimizer']['batch_size'] = 1

    test_net(config, log_loc)

    print('Testing complete')
