import numpy as np

# format [R, Rt, type]

all_experiments = [[4, 4, 'full'], [8, 4, 'full']]

allR = [4,8]
allRt = [1.2, 1.6, 2, 4, 6, 8, 10, 12]
# type = ['ssdu', 'ssdu_bern', 'n2n_unweighted', 'n2n_weighted']
type = ['ssdu_bern']

for i in range(len(type)):
    for j in range(len(allR)):
        for k in range(len(allRt)):
            all_experiments.append([allR[j], allRt[k], type[i]])

print(all_experiments)
print(len(all_experiments))
np.save('all_exp', all_experiments)



