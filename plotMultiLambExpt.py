import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def read_results(log):
    rt = 'logs/cuda/significant_logs_columns_multicoil_6casc/' + log
    file_list_all = os.listdir(rt)

    file_list = file_list_all
    for ii in range(len(file_list_all)):
        if file_list_all[ii][0:8] == log:
            file_list.append(rt + file_list_all[ii])

    loss = []
    singular_loss = []
    RSS_loss = []
    us_fac = []
    SSIM_loss = []
    SSIM_sing_loss = []
    for ii in range(len(file_list)):
        config = load_config(rt + '/' + file_list[ii] + '/config')
        us_fac.append(config['data']['us_fac_lambda'])
        with open(rt + '/' + file_list[ii] + '/test_results.txt') as f:
            lines = f.readlines()
            if len(lines) == 7:
                loss.append(float(lines[1][6:18]))
                SSIM_loss.append(float(lines[2][6:18]))
                singular_loss.append(float(lines[3][15:27]))
                RSS_loss.append(float(lines[4][19:31]))
                SSIM_sing_loss.append(float(lines[5][15:27]))
            else:
                loss.append(float(lines[1][15:27]))
                SSIM_loss.append(float(lines[3][15:27]))

    return loss, singular_loss, RSS_loss, us_fac, SSIM_loss, SSIM_sing_loss

(loss_ssdu, singular_loss, RSS_loss, us_fac_ssdu, SSIM_ssdu, SSIM_sing) = read_results('ssdu_multilambda_4')
(loss_ssdu_p, singular_loss, RSS_loss, us_fac_ssdu_p, SSIM_ssdu_p, SSIM_sing) = read_results('ssdu_prop_multilambda_4')
(loss_unw, singular_loss, RSS_loss, us_fac_unw, SSIM_unw, SSIM_sing) = read_results('n2n_multilambda_4_unweighted')
(loss_w, singular_loss, RSS_loss, us_fac_w, SSIM_w, SSIM_sing) = read_results('n2n_multilambda_4')

print('Best loss is {}, {}, {}'.format(np.min(loss_ssdu), np.min(loss_unw), np.min(loss_w)))
print('Best SSIM is {}, {}, {}'.format(np.max(SSIM_ssdu), np.max(SSIM_unw), np.max(SSIM_w)))

print(SSIM_w, SSIM_ssdu)
print(loss_unw, us_fac_unw)
print(loss_w, us_fac_w)
print(loss_ssdu)
plt.figure()
plt.subplot(121)
#plt.plot([2, 8], [2.694666e-04, 2.694666e-04], '--r')
plt.plot([2, 8], [1.944149e-04, 1.944149e-04], '--r')
plt.plot(us_fac_ssdu, loss_ssdu, 'x', markersize=10)
plt.plot(us_fac_ssdu_p, loss_ssdu_p, '1', markersize=10)
plt.plot(us_fac_unw, loss_unw, '2', markersize=10)
plt.plot(us_fac_w, loss_w, '3', markersize=10)
plt.ylabel('k-space l2 loss')
plt.xlabel('acceleration of second mask')
plt.ylim((0))
#plt.legend(['Fully sampled benchmark',  'SSDU', 'Weighted'])
plt.title('')

plt.subplot(122)
#plt.plot([2, 8], [7.927881e-01, 7.927881e-01 ], '--r')
plt.plot([2, 8], [8.714983e-01, 8.714983e-01], '--r')
plt.plot(us_fac_ssdu, SSIM_ssdu, 'x', markersize=10)
plt.plot(us_fac_ssdu_p, loss_ssdu_p, '1', markersize=10)
plt.plot(us_fac_unw, SSIM_unw, '2', markersize=10)
plt.plot(us_fac_w, SSIM_w, '3', markersize=10)
plt.ylabel('SSIM')
plt.xlabel('acceleration of second mask')
# plt.ylim((0.5, 1))
plt.legend(['Fully sampled benchmark',  'SSDU (bernoulli second mask)', 'SSDU (column second mask)', 'Proposed (Unweighted)', 'Proposed (Weighted)'])
plt.title('')
plt.show()