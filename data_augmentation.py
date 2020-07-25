import numpy as np
from sklearn.decomposition import PCA
import torch


def flip(*arrays):
    horizontal = np.random.random() > 0.5
    vertical = np.random.random() > 0.5
    if horizontal:
        arrays = [np.fliplr(arr) for arr in arrays]
    if vertical:
        arrays = [np.flipud(arr) for arr in arrays]
    return arrays

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def cutout_spatial(self, data):
    cutout_image = np.copy(data)
    x,y = data.shape[:2]
    x_c, y_c = np.random.randint(x, size=2)
    if x_c == x//2 and y_c == y//2:
        return cutout_image
    else:
        cutout_image[x_c, y_c, :] = 0
        return cutout_image

def spatial_combinations(self, data, M=1):
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

def spectral_mean(self, data, M=1):
    new_data = np.copy(data)
    bands = 4*M
    channels = data.shape[-1]
    chunks = channels/bands
    for i in range(math.ceil(chunks)):
        new_data[:,:,int(channels*i/chunks):int(channels*(i+1)/chunks)] = \
        np.stack((np.mean(data[:,:,int(channels*i/chunks):int(channels*(i+1)/chunks)], axis=2) \
        for _ in range(new_data[:,:,int(channels*i/chunks):int(channels*(i+1)/chunks)].shape[-1])), axis=2)
    return new_data

def moving_average(self, data, M=1):
    new_data = np.copy(data)
    channels = data.shape[-1]
    bands = 2*M
    for i in range(channels):
        c1 = np.clip(i-bands, 0, channels)
        c2 = np.clip(i+bands, 0, channels)
        new_data[:,:,i] = np.mean(data[:,:,c1:c2], axis=2)
    return new_data

def spectral_shift(self, data, M=1):
    new_data=np.roll(data, M*2, axis=-1)
    return new_data

def band_combination(self, data, M=1):
    h, w, c = data.shape
    #Test to see if it is possible to use a magnitude as scaling fator for amount of samples to mix from
    size = 2*M
    new_image = np.zeros_like(data)
    splits = c/(M*4)
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
            patch_bands = np.zeros((patch.shape[0], math.ceil(splits), 4*M))
            for i in range(math.ceil(splits)):
                patch_bands[:, i, :] = patch[:,int(c*i/splits):int(c*(i+1)/splits)]
            if patch.shape[0] == 0:
                new_image[x,y,:] = 0
            else:
                for i in range(math.ceil(splits)):
                    rand_band = np.random.randint(patch.shape[0])
                    new_image[x,y,int(c*i/splits):int(c*(i+1)/splits)] = patch_bands[rand_band, i, :]
    return new_image

def identity(data):
    return data

def augment_pool_1():
    augs = [(flip, None, None),
            (radiation_noise, ),
            (spatial_combinations, ),
            (spectral_mean, ),
            (moving_average, ),
            (spectral_shift, ),
            (band_combination, ),
            (identity, None, None)]

def augment_pool_2():
    augs = [(radiation_noise, ),
            (spectral_mean, ),
            (moving_average, ),
            (identity, None, None)]

class RandAugment(object):
    def __init__(self, n, m, patch_size):
        assert n>=1
        assert 1 <= m <= 10

        self.n = n
        self.m = m
        self.patch_size = patch_size
        if self.patch_size > 1:
            self.augment_pool = augment_pool_1()
        else:
            self.augment_pool = augment_pool_2()

    def __call__(self, data):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                data = op(data, v=v, max_v=max_v, bias=bias)
        if self.patch_size > 1:
            data = cutout_spatial(data)
        return data
