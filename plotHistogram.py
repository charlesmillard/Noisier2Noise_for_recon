import os
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import seaborn as sns

sns.set_theme()

log = 'logs/cuda/4.0x/ssdu/65370413_1.2'
res = np.load(log + '/results.npz')

plt.figure()
plt.hist(res['all_ssim'], bins=20)

log = 'logs/cuda/4.0x/ssdu/65370413_2.0'
res = np.load(log + '/results.npz')

plt.figure()
plt.hist(res['all_ssim'], bins=20)


plt.show()


