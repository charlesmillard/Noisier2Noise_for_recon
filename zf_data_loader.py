"""
@author: Charles Millard
"""

import h5py
from os import listdir

from utils import *
from torch.utils.data import Dataset
from mask_tools import *

class zf_data(Dataset):
    def __init__(self, categ='train', config=dict):
        self.__name__ = categ
        self.nx = config['data']['nx']
        self.ny = config['data']['ny']
        self.nc = config['data']['nc']
        self.method = config['data']['method']
        self.sample_type = config['data']['sample_type']
        self.fully_samp_size = config['data']['fully_samp_size']

        self.prob_omega = gen_pdf(self.nx, self.ny, 1 / config['data']['us_fac'], config['data']['poly_order'],
                                  self.fully_samp_size, self.sample_type)
        self.lambda_sample_type = 'bern_inv' if self.method == "ssdu_bern" else self.sample_type
        self.prob_lambda = gen_pdf(self.nx, self.ny, 1 / config['data']['us_fac_lambda'], config['data']['poly_order'],
                                   self.fully_samp_size, self.lambda_sample_type)

        one_minus_eps = 1 - 1e-3
        self.prob_lambda[self.prob_lambda > one_minus_eps] = one_minus_eps
        self.K = torch.as_tensor((1-self.prob_omega)/(1 - self.prob_omega*self.prob_lambda))
        self.K = self.K.unsqueeze(0).unsqueeze(0)

        self.fileRoot = config['data']['location'] + '/multicoil_' + categ + '/'
        self.file_list = listdir(self.fileRoot)

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

        print(categ + ' dataset contains {} slices'.format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        np.random.seed(idx) # same mask at every epoch
        mask_omega = mask_from_prob(self.prob_omega, self.sample_type)

        np.random.seed(None) # different mask at each epoch
        mask_lambda = mask_from_prob(self.prob_lambda, self.lambda_sample_type)

        y0, y, y_tilde = self.get_multicoil(idx, mask_omega, mask_lambda)
        return y0.float(), y.float(), y_tilde.float(), self.K.float()

    def gitem(self, idx):
        return self.__getitem__(idx)

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

        y = mask_omega * y0
        y_tilde = mask_lambda * y

        mx = torch.max(kspace_to_rss(torch.unsqueeze(y, 0)))
        return y0/mx, y/mx, y_tilde/mx
