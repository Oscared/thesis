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

""" It requires to be run in the dataset class as of now...
#PCA augmentation technique. Adds noise in pca space and transform back
def pca_augmentation(pca, data, label, M=1):
    data_aug = np.zeros_like(data)
    data_train = data - np.mean(self.data, axis=(0,1,2))
    for idx, _ in np.ndenumerate(data[:,:,0]):
        x,y = idx
        alpha = M*np.random.uniform(0.9,1.1)
        data_pca = pca.transform(data_train[x,y,:].reshape(1,-1))
        data_pca[:,0] = data_pca[:,0]*alpha
        data_aug[x,y,:] = pca.inverse_transform(data_pca)
    return data_aug
"""

#Cutout augmentation, cut out a random part of the image and replace with ignored label
#this is the vanilla implementation, might not work for this case because
#of the middle pixel. The middle pixel should not be cut out since it is everything
#[0 0 0 0 0]
#[0 0 0 0 0]
#[0 0 P 0 0]
#[0 0 0 0 0]
#[0 0 0 0 0]
#fucked up augmentation purely made for 5x5 patches...
def cutout_spatial(image):
    cutout_image = image
    y = np.random.choice([-1,0,1])
    if y == 0:
        x = np.random.choice([-1,1])
        x_step = 2*x
        x1 = np.min(x, x_step) + 2
        x2 = np.max(x, x_step) + 2

        y_step = np.random.choice([-1,1])
        y1 = np.min(y, y_step) + 2
        y2 = np.max(y, y_step) + 2
    else:
        x = np.random.choice([-1,0,1])
        if x == 0:
            x_step = np.random.choice([-1,1])
            x1 = np.min(x, x_step) + 2
            x2 = np.max(x, x_step) + 2
        else:
            x_step = np.random.choice([-1,1])
            x1 = np.min(x, x_step) + 2
            x2 = np.max(x, x_step) + 2
        y_step = 2*y
        y1 = np.min(y, y_step) + 2
        y2 = np.max(y, y_step) + 2
    cutout_image[y1:y2,x1:x2,:] = 0
    return cutout_image
#Hyperspectral cutout method to cutout part of the spectral bands
def cutout_spectral(image, M=1):
    h, w, channels = image.shape
    #See if magnitude works as factor
    cutouts = 5*M
    bands = 5*M
    new_image = image
    for _ in range(cutouts):
        c = np.random.randint(c)
        c1 = np.clip(c - bands // 2, 0, channels)
        c2 = np.clip(c1 + bands, 0, channels)
        new_image[:,:,c1:c2] = 0
    return new_image

def spatial_combinations(data, M=1):
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
        alphas = np.random.uniform(0.01, 1, size=patch.shape[0])
        new_image[x,y,:] = np.dot(np.transpose(patch), alphas)/np.sum(alphas)
    return new_image

def identity(data):
    return data

def posterize(data, M=1):


def augment_pool():

class RandAugment(object):

    def __init__(self, n, m):
        assert n>=1
        assert 1 <= m <= 10

        self.n = n
        self.m = m
        self.augment_pool = augment_pool()

    def __call__(self, data):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                data = op(data, v=v, max_v=max_v, bias=bias)
        #data = cutout_spatial(data)
        return data
