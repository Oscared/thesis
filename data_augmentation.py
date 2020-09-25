import numpy as np
from sklearn.decomposition import PCA
import random
import math

def flip(data, **kwargs):
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data

def radiation_noise(data, alpha_range=(0.9, 1.1), bias_b=1/25, **kwargs):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + bias_b * noise

def cutout_spatial(data, **kwargs):
    cutout_image = np.copy(data)
    x,y = data.shape[:2]
    x_c, y_c = np.random.randint(x, size=2)
    if x_c == x//2 and y_c == y//2:
        return cutout_image
    else:
        cutout_image[x_c, y_c, :] = 0
        return cutout_image

def spatial_combinations(data, M=1, **kwargs):
    h, w, c = data.shape
    #Test to see if it is possible to use a magnitude as scaling fator for amount of samples to mix from
    size = 2*M
    new_image = np.zeros_like(data)
    for x in range(h):
        for y in range(w):
            x1 = np.clip(x - size // 2, 0, h)
            x2 = np.clip(x + size // 2 + 1, 0, h)
            y1 = np.clip(y - size // 2, 0, w)
            y2 = np.clip(y + size // 2 + 1,  0, w)
            patch = data[x1:x2, y1:y2, :]
            patch = patch.reshape(np.prod(patch.shape[:2]), c)
            delete_idx = []
            for p in range(patch.shape[0]):
                if np.sum(patch[p,:])==0:
                    delete_idx.append(p)
            patch = np.delete(patch, delete_idx, 0)
            if patch.shape[0] == 0:
                new_image[x,y,:] = 0
            else:
                alphas = np.random.uniform(0.01, 1, size=patch.shape[0])
                new_image[x,y,:] = np.dot(np.transpose(patch), alphas)/np.sum(alphas)
    return new_image

def spectral_mean(data, M=1, **kwargs):
    new_data = np.copy(data)
    bands = 4*M
    channels = data.shape[-1]
    chunks = channels/bands
    for i in range(math.ceil(chunks)):
        new_data[:,:,int(channels*i/chunks):int(channels*(i+1)/chunks)] = \
        np.stack((np.mean(data[:,:,int(channels*i/chunks):int(channels*(i+1)/chunks)], axis=2) \
        for _ in range(new_data[:,:,int(channels*i/chunks):int(channels*(i+1)/chunks)].shape[-1])), axis=2)
    return new_data

def moving_average(data, M=1, **kwargs):
    new_data = np.copy(data)
    channels = data.shape[-1]
    bands = 2*M
    for i in range(channels):
        c1 = np.clip(i-bands, 0, channels)
        c2 = np.clip(i+bands, 0, channels)
        new_data[:,:,i] = np.mean(data[:,:,c1:c2], axis=2)
    return new_data

def spectral_shift(data, M=1, **kwargs):
    new_data=np.roll(data, M*2, axis=-1)
    return new_data

def band_combination(data, M=1, **kwargs):
    h, w, c = data.shape
    #Test to see if it is possible to use a magnitude as scaling fator for amount of samples to mix from
    size = 2*M
    new_image = np.zeros_like(data)
    splits = c/(M*4)
    splits_round = math.ceil(splits)
    for x in range(h):
        for y in range(w):
            x1 = np.clip(x - size // 2, 0, h)
            x2 = np.clip(x + size // 2 + 1, 0, h)
            y1 = np.clip(y - size // 2, 0, w)
            y2 = np.clip(y + size // 2 + 1,  0, w)
            patch = data[x1:x2, y1:y2, :]
            patch = patch.reshape(np.prod(patch.shape[:2]), c)
            delete_idx = []
            for p in range(patch.shape[0]):
                if np.sum(patch[p,:])==0:
                    delete_idx.append(p)
            patch = np.delete(patch, delete_idx, 0)
            patch_bands = np.zeros((patch.shape[0], splits_round, 4*M))
            for i in range(splits_round - 1):
                patch_bands[:, i, :] = patch[:,int(c*i/splits):int(c*(i+1)/splits)]
            patch_bands[:, splits_round-1, :] = np.concatenate((patch[:, int(c*(splits_round-1)/splits):], np.zeros((patch.shape[0], int(splits_round*4*M - c)))), axis=1)
            if patch.shape[0]==0 or np.sum(data[x,y,:])==0:
                new_image[x,y,:] = 0
            else:
                for i in range(splits_round - 1):
                    rand_band = np.random.randint(patch.shape[0])
                    new_image[x,y,int(c*i/splits):int(c*(i+1)/splits)] = patch_bands[rand_band, i, :]
                new_image[x,y,int(c*(splits_round-1)/splits):] = patch_bands[rand_band,splits_round-1, :int(c-4*M*(splits_round-1))]
    return new_image

def identity(data, **kwargs):
    return data

def augment_pool_1():
    augs = [(radiation_noise, None, 1/25),
            (spatial_combinations, None, None),
            (spectral_mean, None, None),
            (moving_average, None, None),
            (spectral_shift, None, None),
            (band_combination, None, None),
            (identity, None, None)]
    return augs

def augment_pool_2():
    augs = [(radiation_noise, None, 1/25),
            (spectral_mean, None, None),
            (moving_average, None, None),
            (spectral_shift, None, None),
            (identity, None, None)]
    return augs

def augment_pool_mean():
    augs = [(radiation_noise, None, 1/25),
            (spatial_combinations, None, None),
            (identity, None, None)]
    return augs


class RandAugment(object):
    def __init__(self, n, m, patch_size, special_aug=None):
        assert n>=1
        assert 1 <= m <= 10

        self.n = n
        self.m = m
        self.patch_size = patch_size
        if self.patch_size > 1:
            self.augment_pool = augment_pool_mean()
        else:
            self.augment_pool = augment_pool_2()

        if special_aug is not None:
            self.augment_pool = [(special_aug, None, None)]

    def __call__(self, data):
        #print(data.shape)
        #if self.patch_size == 1:
            #data = np.reshape(data, (1,1,data.shape[0]))
        ops = random.choices(self.augment_pool, k=self.n)
        #print(ops[0][0])
        #print(ops[1][0])
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m+1)
            if random.random() < 0.5:
                data = op(data, M=v, max_v=max_v, bias=bias)
        if self.patch_size > 1:
            data = cutout_spatial(data)
        #else:
            #data = np.reshape(data, (data.shape[2],))
        return data
