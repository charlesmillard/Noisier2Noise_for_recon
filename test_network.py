"""
@author: Charles Millard
"""

from skimage.metrics import structural_similarity as ssim

from varnet_modified import VarNet
from utils import *
from torch.utils.data import DataLoader
from zf_data_loader import zf_data

log_loc = 'logs/cpu/20220711-143909'
print('Testing network saved in ' + log_loc)

config = load_config(log_loc + '/config')
if torch.cuda.is_available():
    config['network']['device'] = 'cuda'
else:
    config['network']['device'] = 'cpu'
    config['optimizer']['batch_size'] = 1

criterion = mse_loss

loss = []
loss_rss = []
all_ssim = []
all_hfen = []
with torch.no_grad():
    base_net = VarNet(num_cascades=config['network']['ncascades'])
    network = pass_varnet

    sd = 380
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

    test_load = DataLoader(zf_data('test', config), batch_size= 2*config['optimizer']['batch_size'], shuffle=True)

    base_net.eval()
    loss = []
    loss_rss = []
    all_ssim = []
    all_hfen = []
    for i, data in enumerate(test_load, 0):
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

        y0_est[pad] = 0

        x0 = kspace_to_rss(y0)
        x0_est = kspace_to_rss(y0_est)
        for ii in range(x0.shape[0]):
            x1 = np.array((x0_est[ii, 0]).detach().cpu())
            x2 = np.array((x0[ii, 0]).detach().cpu())
            all_ssim.append(ssim(x1, x2, data_range=x2.max()))
            loss.append(nmse(np.asarray(y0_est[ii].detach().cpu()), np.asarray(y0[ii].detach().cpu())))
            loss_rss.append(nmse(x1, x2))

        break

f = open(log_loc + "/test_results.txt", 'w')
f.write("model location is " + log_loc + "\n")
f.write("loss: {:e} \n".format(10*np.log10(np.mean(loss).item())))
f.write("RSS loss: {:e} \n".format(10*np.log10(np.mean(loss_rss).item())))
f.write("SSIM: {:e} \n".format(np.mean(all_ssim)))
f.write("HFEN: {:e} \n".format(10*np.log10(np.mean(all_hfen))))
f.write("parameters in model: {:e} \n".format(pp))

np.savez(log_loc + '/results.npz', loss=loss, loss_rss=loss_rss, all_ssim=all_ssim, all_hfen=all_hfen)

print('Text file with results saved in ' + log_loc)