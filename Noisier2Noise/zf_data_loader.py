"""
@author: Charles Millard
"""

import h5py
from os import listdir

from Noisier2Noise.utils import *
from torch.utils.data import Dataset
from Noisier2Noise.mask_tools import *


class subSampledData(Dataset):
    def __init__(self, categ='train', config=dict):
        self.__name__ = categ
        self.nx = config['data']['nx'] #640
        self.ny = config['data']['ny'] #320
        self.nc = config['data']['nc'] #15 以上三个变量实现了统一大小
        self.method = config['data']['method']
        self.sample_type = config['data']['sample_type']
        self.fully_samp_size = config['data']['fully_samp_size']

        # generate probability maps
        self.prob_omega = gen_pdf(self.nx, self.ny, 1 / config['data']['us_fac'], config['data']['poly_order'],
                                  self.fully_samp_size, self.sample_type)
        self.lambda_sample_type = config['data']['sample_type_lambda']
        self.prob_lambda = gen_pdf(self.nx, self.ny, 1 / config['data']['us_fac_lambda'], config['data']['poly_order'],
                                   self.fully_samp_size, self.lambda_sample_type)

        one_minus_eps = 1 - 1e-3
        self.prob_lambda[self.prob_lambda > one_minus_eps] = one_minus_eps

        # compute K
        self.K = torch.as_tensor((1 - self.prob_omega) / (1 - self.prob_omega * self.prob_lambda))
        self.K = self.K.unsqueeze(0).unsqueeze(0)


        self.fileRoot = config['data']['location'] + '/multicoil_' + categ + '/'
        self.file_list = listdir(self.fileRoot)

        print('counting slices...')
        file_list_corrected = []
        nslices = []
        for file_idx in range(len(self.file_list)):
            file = self.file_list[file_idx]
            f = h5py.File(self.fileRoot + file, 'r')
            if f['kspace'].shape[1] == self.nc: # only include if contains nc coils
                file_list_corrected.append(file)
                nslices.append(f['kspace'].shape[0]) #第0个维度(number of slices, number of coils, height, width).
        self.file_list = file_list_corrected
        self.slice_cumsum = np.cumsum(nslices) #计算总的slice的数量，适用于不定长度 不定元素大小的元素序列求和
        self.slice_cumsum = np.insert(self.slice_cumsum, 0, 0) #增加一个维度
        self.len = int(np.sum(nslices)) #总的slice的个数 直接求和就好了

        print(categ + ' dataset contains {} slices'.format(self.len))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        np.random.seed(idx)  # same mask at every epoch 数据间不同 数据内每个epoch相同
        mask_omega = mask_from_prob(self.prob_omega, self.sample_type)

        np.random.seed(None)  # different mask at each epoch
        mask_lambda = mask_from_prob(self.prob_lambda, self.lambda_sample_type)

        y0, y, y_tilde = self.get_multicoil(idx, mask_omega, mask_lambda)
        
        return y0.float(), y.float(), y_tilde.float(), self.K.float()

    def get_multicoil(self, idx, mask_omega, mask_lambda):
        file_idx = np.where(idx >= self.slice_cumsum)[0][-1] # 输出self.slice_cumsum中满足条件的索引位置 可以得到按病人的idx 第几个idx对应的slice是病人分界线
        slice_idx = idx - self.slice_cumsum[file_idx] #按病人计数的idx 是一个list

        file = self.file_list[file_idx]
        f = h5py.File(self.fileRoot + file, 'r')
        y0 = torch.as_tensor(f['kspace'][slice_idx])
        y0 = torch.flip(y0, [1]) #按照维度对输入进行翻转
        y0 = torch.permute(torch.view_as_real(y0), (3, 0, 1, 2))

        y0 = pad_or_trim_tensor(y0, self.nx, self.ny)

        mask_omega = torch.as_tensor(mask_omega).unsqueeze(0).unsqueeze(0).int() #under mask
        mask_lambda = torch.as_tensor(mask_lambda).unsqueeze(0).unsqueeze(0).int() #loss mask

        y = mask_omega * y0
        y_tilde = mask_lambda * y

        mx = torch.max(kspace_to_rss(torch.unsqueeze(y, 0))) # normalization

        return y0 / mx, y / mx, y_tilde / mx
