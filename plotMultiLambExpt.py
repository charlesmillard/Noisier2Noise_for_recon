import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import seaborn as sns

all_net_logs = []
for l in os.walk('logs/cuda/4.0x'):
    if l[0][-1] not in ['u', 'n', 'd', 'x', 's', 'l']:
        all_net_logs.append(l[0])
for q in os.walk('logs/cuda/8.0x'):
    if q[0][-1] not in ['u', 'n', 'd', 'x', 's', 'l']:
        all_net_logs.append(q[0])

mean_ssim = []
std_ssim = []
mean_l2 = []
std_l2 = []
mean_l2_rss = []
std_l2_rss = []
train_type = []
us_fac = []
us_fac_lambda = []
weighted = []
for log in all_net_logs:
    if os.path.exists(log + '/results.npz'):
        res = np.load(log + '/results.npz')
        mean_l2.append(np.mean(res['loss']))
        std_l2.append(np.std(res['loss']))
        mean_l2_rss.append(np.mean(res['loss_rss']))
        std_l2_rss.append(np.std(res['loss_rss']))
        mean_ssim.append(np.mean(res['all_ssim']))
        std_ssim.append(np.std(res['all_ssim']))

        config = load_config(log + '/config')
        train_type.append(config['data']['method'])
        us_fac.append(config['data']['us_fac'])
        us_fac_lambda.append(config['data']['us_fac_lambda'])
        weighted.append(config['optimizer']['weight_loss'])

sns.set_theme()
col = ['black', 'red', 'green', 'blue', 'orange']

# marker = ['--', 'o-', '*-', 's-', 'p-']
marker = ['-', '-', '-', '-', '-']

stats_of_int =[mean_l2]
stds_of_int =[std_l2,  std_ssim]
names = ['NMSE', 'SSIM']
ylims = [[0.11, 0.18], [0.14, 0.2]]
s_best = []

lim_idx = 0
for us in [4, 8]:
    plt.figure(figsize=(5, 3))
    plt.rcParams.update({'font.family': 'Linux Libertine Display O'})
    # plt.rcParams.update({"font.family": "Helvetica"})
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    for ii in range(len(stats_of_int)):

        s_best.append([])
        plt.subplot(1, len(stats_of_int), ii+1)
        s = stats_of_int[ii]
        sd = stds_of_int[ii]
        max_or_min = np.min if ii in [0] else np.max
        idx = [i for i in range(len(train_type)) if train_type[i] == 'full' and us_fac[i] == us]
        plt.plot([1, 12], [s[idx[0]], s[idx[0]]], marker[0], markersize=10, color=col[0])
        s_best[ii].append(s[idx[0]])

        idx_all = [[i for i in range(len(train_type)) if train_type[i] == 'ssdu_bern' and us_fac[i] == us],
                   [i for i in range(len(train_type)) if train_type[i] == 'ssdu' and us_fac[i] == us],
                   [i for i in range(len(train_type)) if
                    train_type[i] == 'n2n' and us_fac[i] == us and weighted[i] == False],
                   [i for i in range(len(train_type)) if
                    train_type[i] == 'n2n' and us_fac[i] == us and weighted[i] == True]]
        for jj in range(len(idx_all)):
            idx = idx_all[jj]
            unordered_us = [us_fac_lambda[i] for i in idx]
            sorted_idx = [idx[k] for k in sorted(range(len(unordered_us)), key=lambda k: unordered_us[k])]
            # plt.errorbar([us_fac_lambda[i] for i in sorted_idx], [s[i] for i in sorted_idx], yerr=[sd[i] for i in sorted_idx], markersize=10)
            plt.plot([us_fac_lambda[i] for i in sorted_idx], [s[i] for i in sorted_idx], marker[jj+1], markersize=5, color=col[jj + 1])
            s_best[ii].append(max_or_min([s[i] for i in idx]) if idx != [] else 0)
        # plt.errorbar([us_fac_lambda[i] for i in idx], [s[i] for i in idx], yerr=[sd[i] for i in idx], markersize=10)
        plt.xlabel('$\widetilde{R}$')
        plt.ylabel(names[ii])
        plt.ylim(ylims[lim_idx])
        plt.xticks([2, 4, 6, 8, 10, 12])
        lim_idx += 1

    plt.legend(['Fully supervised', 'SSDU (original)', 'SSDU (proposed)', 'Noisier2Noise (unweighted)', 'Noisier2Noise (weighted)'], loc='upper right', fontsize=8, ncol=5)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95, wspace=0.25, hspace=0.2)
# plt.show()

plt.show()
print(s_best)