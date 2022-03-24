import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from utils import ifftnc

if os.path.isdir('/home/fs0/xsd618/'):  # jalapeno
    root = '/home/fs0/xsd618/scratch/fastMRI_brain/'
elif os.path.isdir('/well/chiew/users/fjv353/'):  # rescomp
    root = '/well/chiew/users/fjv353/fastMRI_brain/'
else:  # my laptop
    root = '/home/user/data/fastMRI_test_subset/'

categories = ['train', 'val', 'test']

for categ in categories:
    smapsRoot = root + 'multicoil_' + categ + '_smaps/'
    fileRoot = root + 'multicoil_' + categ + '/'

    saveTo = root + 'single_coil_simulation/' + categ + '/'


    file_list = os.listdir(fileRoot)
    print(file_list)
    nslices = [0]*len(file_list)
    for file_idx in range(len(file_list)):
        file = file_list[file_idx]
        f = h5py.File(fileRoot + file, 'r')
        nslices[file_idx] = f['kspace'].shape[0]

    slice_cumsum = np.cumsum(nslices)
    slice_cumsum = np.insert(slice_cumsum, 0, 0)

    total_slices = np.sum(nslices)

    for idx in range(total_slices):
        file_idx = np.where(idx >= slice_cumsum)[0][-1]
        slice_idx = idx - slice_cumsum[file_idx]

        file = file_list[file_idx]
        f = h5py.File(fileRoot + file, 'r')
        dcoil_full = np.transpose(f['kspace'][slice_idx:slice_idx + 1], (0, 3, 2, 1))
        dcoil_full = np.squeeze(dcoil_full)

        smap_name = smapsRoot + file[:-3] + '_sl' + str(slice_idx) + '.npy'
        smaps = np.squeeze(np.load(smap_name))

        x0 = np.conj(smaps) * ifftnc(dcoil_full)
        x0 = np.sum(x0, axis=2)

        xnorm = np.max(np.abs(x0))
        x0 /= xnorm

        x0 = x0.astype('csingle')

        np.save(saveTo + 'file' + str(file_idx) + '_slice' + str(slice_idx) + '_x0', x0)

        # print(idx)