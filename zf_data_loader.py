import torch
import h5py
import sys
import matplotlib.pyplot as plt
import os

from utils import *
from torch.utils.data import Dataset
from os import listdir
from mask_tools import *

class zf_data(Dataset):
    def __init__(self, categ='train', config=dict):
        self.__name__ = categ
        self.nx = config['data']['nx']
        self.ny = config['data']['ny']
        self.method = config['data']['method']
        self.sample_type = config['data']['sample_type']
        self.multicoil = config['data']['multicoil']
        self.nc = config['data']['nc'] if self.multicoil else 1
        if torch.cuda.is_available(): self.nc = 16
        self.fully_samp_size = config['data']['fully_samp_size']

        self.prob_omega = genPDF(self.nx, self.ny, 1 / config['data']['us_fac'], config['data']['poly_order'], self.fully_samp_size, self.sample_type)
        # self.lambda_sample_type = 'bern' if self.method == "ssdu" else self.sample_type
        self.lambda_sample_type = self.sample_type
        self.prob_lambda = genPDF(self.nx, self.ny, 1 / config['data']['us_fac_lambda'], config['data']['poly_order'], self.fully_samp_size, self.lambda_sample_type)

        one_minus_eps = 1 - 1e-3
        self.prob_lambda[self.prob_lambda > one_minus_eps] = one_minus_eps

        self.K = torch.as_tensor((1-self.prob_omega)/(1 - self.prob_omega*self.prob_lambda))
        self.K = self.K.unsqueeze(0).unsqueeze(0) if self.multicoil else self.K.unsqueeze(0)

        print('K in corner of k-space is {}'.format(self.K[0, 0, 0, 0]))

        if os.path.isdir('/home/fs0/xsd618/'): # jalapeno
            if self.multicoil:
                root = '/home/fs0/xsd618/scratch/fastMRI_brain/multicoil_'
            else:
                root = 0
        elif os.path.isdir('/well/chiew/users/fjv353/'): # rescomp
            if self.multicoil:
                root = '/well/chiew/users/fjv353/fastMRI_brain/multicoil_'
            else:
                root = '/well/chiew/users/fjv353/fastMRI_brain/single_coil_simulation/'
        else: # my laptop
            if self.multicoil:
                root = '/home/xsd618/data/fastMRI_test_subset_brain/multicoil_'
            else:
                root = '/home/xsd618/data/fastMRI_test_subset/single_coil_simulation/'

        self.fileRoot = root + categ + '/'
        self.file_list = listdir(self.fileRoot)

        if self.multicoil:
            print('counting slices...')
            file_list_corrected = []
            nslices = []
            for file_idx in range(len(self.file_list)):
                file = self.file_list[file_idx]
                f = h5py.File(self.fileRoot + file, 'r')
                if f['kspace'].shape[1] == self.nc:
                    file_list_corrected.append(file)
                    nslices.append(f['kspace'].shape[0])
            self.file_list = file_list_corrected
            self.slice_cumsum = np.cumsum(nslices)
            self.slice_cumsum = np.insert(self.slice_cumsum, 0, 0)
            self.len = int(np.sum(nslices))
        else:
            self.len = len(self.file_list)

        dataset_trunc = config['data']['train_trunc'] if categ == 'train' else config['data']['val_trunc']
        if len(self.file_list) > dataset_trunc:
            self.file_list = self.file_list[:dataset_trunc]
        print(categ + ' dataset contains {} slices'.format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        np.random.seed(idx) # same mask at every epoch
        mask_omega = maskFromProb(self.prob_omega, self.sample_type)

        np.random.seed(None)
        mask_lambda = maskFromProb(self.prob_lambda, self.lambda_sample_type)

        if self.multicoil:
            (y0, y, y_tilde) = self.get_multicoil(idx, mask_omega, mask_lambda)
        else:
            (y0, y, y_tilde) = self.get_singlecoil(idx, mask_omega, mask_lambda)

        return y0.float(), y.float(), y_tilde.float(), self.K.float()

    def gitem(self, idx):
        return self.__getitem__(idx)

    def get_singlecoil(self, idx, mask_omega, mask_lambda, mask_theta):
        file = self.file_list[idx]
        x0 = np.load(self.fileRoot + file)
        x0 = np.rot90(x0)
        x0 = pad_or_trim(x0, self.nx, self.ny)
        y0 = fftnc(x0)

        np.random.seed(idx) # same mask at every epoch
        y = mask_omega*y0

        np.random.seed(None) # different Lambda mask
        if self.method == "ssdu":
            y = mask_lambda*y0
            y_tilde = mask_theta*y0
        else:
            y_tilde = mask_lambda*y

        y0 = torch.view_as_real(torch.as_tensor(y0))
        y = torch.view_as_real(torch.as_tensor(y))
        y_tilde = torch.view_as_real(torch.as_tensor(y_tilde))

        y0 = torch.permute(y0, (2, 0, 1))
        y = torch.permute(y, (2, 0, 1))
        y_tilde = torch.permute(y_tilde, (2, 0, 1))

        return y0, y, y_tilde

    def get_multicoil(self, idx, mask_omega, mask_lambda):
        file_idx = np.where(idx >= self.slice_cumsum)[0][-1]
        slice_idx = idx - self.slice_cumsum[file_idx]

        file = self.file_list[file_idx]
        f = h5py.File(self.fileRoot + file, 'r')
        y0 = torch.as_tensor(f['kspace'][slice_idx])
        y0 = torch.flip(y0, [1])
        y0 = torch.permute(torch.view_as_real(y0), (3, 0, 1, 2))

        y0 = pad_or_trim_tensor(y0, self.nx, self.ny)

        mask_omega = torch.as_tensor(mask_omega).unsqueeze(0).unsqueeze(0).int()
        mask_lambda = torch.as_tensor(mask_lambda).unsqueeze(0).unsqueeze(0).int()

        y = mask_omega*y0
        y_tilde = mask_lambda * y

        mx = torch.max(kspaceToRSS(torch.unsqueeze(y, 0)))
        return y0/mx, y/mx, y_tilde/mx
