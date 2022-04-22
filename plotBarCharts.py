import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

type = 'test'
root = 'logs/cuda/significant_logs_columns_multicoil_6casc/8x/'
if type == '4':
    log_loc = [root + 'full_4', root + 'ssdu_multilambda_4/60008780_lambda_us_2',
               root + 'ssdu_prop_multilambda_4/61125448_lambda_us_20',
               root + 'n2n_multilambda_4_unweighted/60213858_lambda_us_8',
               root + 'n2n_multilambda_4/59623615_lambda_us_8']
elif type == '8':
    log_loc = [root + 'full_8', root + 'ssdu_multilambda_8/60258597_lambda_us_12',
               root + 'ssdu_prop_multilambda_8/61031656_lambda_us_20',
               root + 'n2n_multilambda_8_unweighted/60214892_lambda_us_8',
               root + 'n2n_multilambda_8/59626802_lambda_us_8']
elif type == '4m_unw':
    log_loc = [root + 'n2n_multilambda_4_unweighted/60213858_lambda_us_2',
               root + 'n2n_multilambda_4_unweighted/60213858_lambda_us_4',
               root + 'n2n_multilambda_4_unweighted/60213858_lambda_us_6',
               root + 'n2n_multilambda_4_unweighted/60213858_lambda_us_8']
elif type == '8m_unw':
    log_loc = [root + 'n2n_multilambda_8_unweighted/60214892_lambda_us_2',
               root + 'n2n_multilambda_8_unweighted/60214892_lambda_us_4',
               root + 'n2n_multilambda_8_unweighted/60214892_lambda_us_6',
               root + 'n2n_multilambda_8_unweighted/60214892_lambda_us_8']
elif type == '4m':
    log_loc = [root + 'n2n_multilambda_4/59623615_lambda_us_8', root + 'n2n_multilambda_4/59623615_lambda_us_8',
               root + 'n2n_multilambda_4/59623615_lambda_us_8', root + 'n2n_multilambda_4/59623615_lambda_us_8']
elif type == '8m':
    log_loc = [root + 'n2n_multilambda_8/59626802_lambda_us_2', root + 'n2n_multilambda_8/59626802_lambda_us_4',
               root + 'n2n_multilambda_8/59626802_lambda_us_6', root + 'n2n_multilambda_8/59626802_lambda_us_8']
elif type == 'test':
    log_loc = [root + 'n2n_multilambda_8/59626802_lambda_us_80/cpu_examples', root + 'n2n_multilambda_8/59626802_lambda_us_80/cpu_examples',
               root + 'n2n_multilambda_8/59626802_lambda_us_80/cpu_examples', root + 'n2n_multilambda_8/59626802_lambda_us_80/cpu_examples',
               root + 'n2n_multilambda_8/59626802_lambda_us_80/cpu_examples']

names = ['Fully supervised', 'Noisier2Noise (weighted)', 'Noisier2Noise (unweighted)', 'SSDU (original)', 'SSDU (proposed)']

loss = []
RSS_loss = []
SSIM = []
HFEN = []

for l in log_loc:
    with open(l + '/test_results.txt') as f:
        lines = f.readlines()
        loss.append(float(lines[1][6:18]))
        RSS_loss.append(float(lines[2][10:22]))
        SSIM.append(float(lines[3][6:18]))
        HFEN.append(float(lines[4][6:18]))

print(loss)
sns.set_theme()

col = ['black', 'red', 'green', 'blue', 'cyan']
lab = [' ', '  ', '   ', '    ', '      ']

x_pos = [0, 0.5, 1, 1.5, 2.0, 2.5]
wd = 0.9

plt.figure(figsize=(5,3))
plt.grid()
plt.rcParams.update({'font.family':'Linux Libertine Display O'})
plt.subplot(221)
plt.ylabel('NMSE, k-space')
plt.bar(lab, loss, color=col, width=wd)

plt.subplot(222)
plt.ylabel('NMSE, image-domain')
plt.bar(lab,  RSS_loss, color=col, width=wd)

plt.subplot(223)
plt.ylabel('SSIM')
for ii in range(len(loss)):
    plt.bar(lab[ii], HFEN[ii], color=col[ii], width=wd)

# plt.subplot(224)
# plt.ylabel('HFEN')
# for ii in range(len(loss)):
#     plt.bar(lab[ii], HFEN[ii], color=col[ii], width=1)

plt.legend(names, framealpha=1)
plt.subplots_adjust(left=0.2,
                    bottom=0.05,
                    right=0.95,
                    top=0.95,
                    wspace=0.5,
                    hspace=0.2)
plt.savefig("bar_charts/bar_chart_" + type +  ".eps")
plt.show()