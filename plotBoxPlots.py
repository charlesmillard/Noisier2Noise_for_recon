import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils import load_config
import matplotlib.patches as mpatches

type = '4and8'
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
elif type == '4and8':
    root4 = 'logs/cuda/4.0x/'
    root8 = 'logs/cuda/8.0x/'
    log_loc = [root4 + 'full/66595042_4.0', root4 + 'ssdu_bern/66595042_1.6',
               root4 + 'ssdu/65370413_1.6',
               root4 + 'n2n_unweighted/65370413_4.0',
               root4 + 'n2n_weighted/65370413_4.0', root8 + 'full/66595042_4.0', root8 + 'ssdu_bern/66595042_2.0',
               root8 + 'ssdu/65370413_2.0',
               root8 + 'n2n_unweighted/65370413_10.0',
               root8 + 'n2n_weighted/65370413_10.0']

all_nmse = []
all_ssim = []
ii = 0
for log in log_loc:
    if os.path.exists(log + '/results.npz'):
        res = np.load(log + '/results.npz')
        all_nmse.append(res['loss'])
        all_ssim.append(res['all_ssim'])
        ii+=1
        if ii==5:
            all_nmse.append([])
            all_ssim.append([])


sns.set_theme()
col = ['black', 'red', 'green', 'blue', 'orange', 'black', 'black', 'red', 'green', 'blue', 'orange']
col2 = ['black','black', 'red', 'red', 'green', 'green', 'blue', 'blue', 'orange', 'orange', 'black', 'black']*2

fig = plt.figure(figsize=(6.5, 9))
plt.rcParams.update({'font.family': 'Linux Libertine Display O'})
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
ax = fig.add_subplot(211)
bp = ax.boxplot(all_nmse,
                notch ='False', showfliers=False)

ii = 0
for median in bp['medians']:
    median.set(color =col[ii], linewidth = 3)
    ii+=1

# ax.set_xticklabels(['Fully\nsupervised', 'SSDU\n(original)', 'SSDU\n(proposed)', 'Noisier2Noise\n(unweighted)', 'Noisier2Noise\n(weighted)']*2)
ax.set_xticklabels(['', '', '$R=4$', '', '', '', '','', '$R=8$',  '', ''])
ax.set_ylabel('NMSE',  fontsize=20)
ax.set_ylim([0.0, 0.36])
nm = ['Fully supervised', 'SSDU (original)', 'SSDU (proposed)', 'Noisier2Noise (unweighted)', 'Noisier2Noise (weighted)']
hand = []
for ii in range(5):
    hand.append(mpatches.Patch(color=col[ii], label=nm[ii]))
plt.legend(handles=hand, ncol=2, fontsize=10, loc='upper left')
##
ax = fig.add_subplot(212)
bp = ax.boxplot(all_ssim,
                notch ='False', showfliers=False)
ax.set_ylim([0.55, 1])

ii = 0
for median in bp['medians']:
    median.set(color =col[ii], linewidth = 3)
    ii+=1

# ii = 0
# for median in bp['whiskers']:
#     median.set(color=col2[ii], linewidth=2)
#     ii+=1
# ii = 0
# for median in bp['caps']:
#     median.set(color=col2[ii], linewidth=2)
#     ii += 1
# ii = 0
# for median in bp['boxes']:
#     median.set(color=col[ii], linewidth=1)
#     ii+=1
# ii = 0
# for median in bp['medians']:
#     median.set(color=col[ii], linewidth=1)
#     ii+=1

ax.set_xticklabels(['', '', '$R=4$', '', '', '', '', '', '$R=8$', '', ''])
ax.set_ylabel('SSIM',  fontsize=20)

plt.subplots_adjust(left=0.13, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.1)




# plt.legend(['Fully\nsupervised', 'SSDU\n(original)', 'SSDU\n(proposed)', 'Noisier2Noise\n(unweighted)', 'Noisier2Noise\n(weighted)'], loc='upper right', fontsize=8, ncol=5)
plt.show()