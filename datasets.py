import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.decomposition import PCA
import utils
import math

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils import open_file

dataset_path = "/data/"

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder=dataset_path):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to /data/
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    folder = target_folder

    if not os.path.isdir(folder):
        os.mkdir(folder)

    if dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'Salinas':
        # Load the image
        img = open_file(folder + 'Salinas_corrected.mat')
        img = img['salinas_corrected']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Salinas_gt.mat')['salinas_gt']
        label_values = ["Undefined", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow",
                        "Fallow_rough_plow", "Fallow_smooth", "Stubble",
                        "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                        "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
                        "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained",
                        "Vinyard_vertical_trellis"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaC':
        img = open_file(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        #gt = np.zeros(img.shape)
        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = None

        ignored_labels = [0]
    else:
        print("Error: no dataset of the requested type found. Available datasets are PaviaU, Salinas.")

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
    #data = preprocessing.scale(data)
    data  = preprocessing.minmax_scale(data)
    img = data.reshape(img.shape)

    return img, gt, label_values, ignored_labels, rgb_bands, palette


def get_patch_data(dataset_name, patch_size, target_folder=dataset_path, fold=0):
    """ Get data from patch based extraction from Nalepa et al
    Args:
        dataset_name: string with the name of the dataset (salinas, pavia, indiana)
        patch_size: size of patches to extract
        target_folder (optional): folder to store the datasets, defaults to /data/
        fold (optional): which fold of the split to use, defaults to 0 (0-4 for salinas and pavia, 0-3 for indiana)
    Returns:
        train_img: list of 3D hyperspectral image patches (PxWxHxB) for training
        train_gt: list of 2D int array of labels for training
        test_patch: 3D hyperspectral image (WxHxB) for testing
        test_gt: list of 2D int array of labels for testing
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    train_patches = []
    train_patches_gt = []

    dataset_name = dataset_name.lower()

    if dataset_name not in ['salinas', 'pavia', 'indiana']:
        print('Error: Dataset is not available')

    if dataset_name == 'pavia':
        remove_files = 4
    else:
        remove_files = 3

    for i in (range(int((len(os.listdir(target_folder + '/{}_fold_{}/'.format(dataset_name, fold)))-remove_files)/2))):
        #train_patches[i] = np.load(data_path + '/salinas_fold_0/patch_{}.npy'.format(i))
        #train_gt[i] = np.load(data_path + '/salinas_fold_0/patch_{}_gt.npy'.format(i))

        train_patches.append(np.load(target_folder + '/{}_fold_{}/patch_{}.npy'.format(dataset_name, fold, i)))
        train_patches_gt.append(np.load(target_folder + '/{}_fold_{}/patch_{}_gt.npy'.format(dataset_name, fold, i)))

    test_img = np.load(target_folder + '/{}_fold_{}/test.npy'.format(dataset_name, fold))
    test_img_gt = np.load(target_folder + '/{}_fold_{}/test_gt.npy'.format(dataset_name, fold))

    #Normalize both with the training set to get [0,1] on training and use same normalization on test (whole image/dataset, not per band)
    min = np.min(train_patches)
    max = np.max(train_patches)
    train_patches = (train_patches - min)/(max - min)
    test_img = (test_img - min)/(max - min)

    pad_width = patch_size // 2

    train_img = np.pad(train_patches, ((0,0), (pad_width, pad_width), (pad_width, pad_width), (0,0)))
    train_gt = np.pad(train_patches_gt, ((0,0), (pad_width, pad_width), (pad_width, pad_width)))
    test_patch = np.pad(test_img, ((pad_width, pad_width), (pad_width, pad_width), (0,0)))
    test_gt = np.pad(test_img_gt, ((pad_width, pad_width), (pad_width, pad_width)))

    train_img = np.asarray(train_img, dtype='float32')
    test_patch = np.asarray(test_patch, dtype='float32')
    #Normalize test and train
    """
    #This scales all bands seperatly, results in high values of some bands in test to be large
    scaler = preprocessing.MinMaxScaler()
    data = train_img.reshape(np.prod(train_img.shape[:3]), np.prod(train_img.shape[3:]))
    data = scaler.fit_transform(data)
    train_img = data.reshape(train_img.shape)

    data_test = test_patch.reshape(np.prod(test_patch.shape[:2]), np.prod(test_patch.shape[2:]))
    data_test = scaler.transform(data_test)
    test_patch = data_test.reshape(test_patch.shape)
    """

    if dataset_name == 'pavia':
        rgb_bands = (55, 41, 12)
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = [0]

    elif dataset_name == 'salinas':
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        label_values = ["Undefined", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow",
                        "Fallow_rough_plow", "Fallow_smooth", "Stubble",
                        "Celery", "Grapes_untrained", "Soil_vinyard_develop",
                        "Corn_senesced_green_weeds", "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
                        "Lettuce_romaine_6wk", "Lettuce_romaine_7wk", "Vinyard_untrained",
                        "Vinyard_vertical_trellis"]
        ignored_labels = [0]

    elif dataset_name == 'indiana':
        rgb_bands = (43, 21, 11)  # AVIRIS sensor
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]
    else:
        print("Error: no dataset of the requested type found. Available datasets are PaviaU, Salinas.")

    return train_img, train_gt, test_patch, test_gt, label_values, ignored_labels, rgb_bands, palette

    """
    train_img = []
    train_gt = []

    kwargs = {'step': 1, 'window_size': (patch_size, patch_size), 'with_data': True}

    for i in range(len(train_patches)):
        for data, x,y,_,_ in enumerate(utils.sliding_window(train_patches[i], **kwargs)):

    """

class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, labeled=True, **args):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.patch_size = args['patch_size']
        self.name = args['dataset']
        self.ignored_labels = set(args['ignored_labels'])
        self.flip_augmentation = args['flip_augmentation']
        self.radiation_augmentation = args['radiation_augmentation']
        self.mixture_augmentation = args['mixture_augmentation']
        self.center_pixel = args['center_pixel']
        self.labeled = labeled
        self.pca_aug = args['pca_augmentation']
        supervision = args['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        #Note: added that x and y could equal p as well since positions on array are zero indexed
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x >= p and x < data.shape[0] - p and y >= p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        self.indices_shuffle = np.copy(self.indices)
        np.random.shuffle(self.indices_shuffle)

        self.class_var = {}
        for c in np.unique(self.label):
            if c not in self.ignored_labels:
                l_indices = np.nonzero(self.labels==c)
                pos = self.indices[l_indices]
                var = np.var(self.data[pos[:,0], pos[:,1]], axis=0)
                self.class_var[c] = np.diag(var)


        centered_data = self.data - np.mean(self.data)
        data_train, _ = utils.build_dataset(centered_data, self.label, ignored_labels = self.ignored_labels)
        self.pca = PCA(n_components=11)
        self.pca.fit(data_train)


    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                #This is the original implementation, but I think it takes the wrong data
                #x, y = self.indices_shuffle[l_indice]
                #This is the new implementaiton, it does not mix indices and should take the right sample
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    #PCA augmentation technique. Adds noise in pca space and transform back
    def pca_augmentation(self, data, label, M=1):
        data_aug = np.zeros_like(data)
        data_train = data - np.mean(self.data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                x,y = idx
                alpha = M*np.random.uniform(0.9,1.1)
                data_pca = self.pca.transform(data_train[x,y,:].reshape(1,-1))
                data_pca[:,0] = data_pca[:,0]*alpha
                data_aug[x,y,:] = self.pca.inverse_transform(data_pca)
        return data_aug

    #Cutout augmentation, cut out a random part of the image and replace with ignored label
    #this is the vanilla implementation, might not work for this case because
    #of the middle pixel. The middle pixel should not be cut out since it is everything
    #[0 0 0 0 0]
    #[0 0 0 0 0]
    #[0 0 P 0 0]
    #[0 0 0 0 0]
    #[0 0 0 0 0]
    #fucked up augmentation purely made for 5x5 patches...
    def cutout_hsi(image):
        cutout_image = image
        y = np.random.choice([-1,0,1])
        if y == 0:
            x = np.random.choice([-1,1])
            x_step = 2*x
            x1 = np.min(x, x_step)
            x2 = np.max(x, x_step)

            y_step = np.random.choice([-1,1])
            y1 = np.min(y, y_step)
            y2 = np.max(y, y_step)
        else:
            x = np.random.choice([-1,0,1])
            if x == 0:
                x_step = np.random.choice([-1,1])
                x1 = np.min(x, x_step)
                x2 = np.max(x, x_step)
            else:
                x_step = 2*x
                x1 = np.min(x, x_step)
                x2 = np.max(x, x_step)
            y_step = 2*y
            y1 = np.min(y, y_step)
            y2 = np.max(y, y_step)
        cutout_image[y1:y2,x1:x2,:] = 0
        return cutout_image
    def cutout(image, size=2, n_squares=1):
        h, w, channels = image.shape
        new_image = image
        for _ in range(n_squares):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - size // 2, 0, h)
            y2 = np.clip(y + size // 2, 0, h)
            x1 = np.clip(x - size // 2, 0, w)
            x2 = np.clip(x + size // 2, 0, w)
            new_image[y1:y2,x1:x2,:] = 0
        return new_image

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices_shuffle[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        """
        If the dataset is labeled return data accoring with data augmentation
        and the ground truth of that specific data.
        """
        if self.labeled == True:
            data = self.data[x1:x2, y1:y2]
            label = self.label[x1:x2, y1:y2]

            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                data, label = self.flip(data, label)
            if self.radiation_augmentation and np.random.random() < 0.1:
                data = self.radiation_noise(data)
            if self.mixture_augmentation and np.random.random() < 0.2:
                data = self.mixture_noise(data, label)
            if self.pca_aug and np.random.random() < 0.2:
                data = self.pca_augmentation(data)

            # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
            data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
            label = np.asarray(np.copy(label), dtype='int64')

            # Load the data into PyTorch tensors
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            # Extract the center label if needed
            if self.center_pixel and self.patch_size > 1:
                label = label[self.patch_size // 2, self.patch_size // 2]
            # Remove unused dimensions when we work with invidual spectrums
            elif self.patch_size == 1:
                data = data[:, 0, 0]
                label = label[0, 0]

            # Add a fourth dimension for 3D CNN
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                data = data.unsqueeze(0)
            return data, label

        """
        If the data is unlabeled, return one version of the data with weak
        data augmentation and one version of the same data with strong
        data augmentation.
        """
        if self.labeled == False:
            data_weak = self.data[x1:x2, y1:y2]
            data_strong = np.copy(data_weak)
            label_weak = self.label[x1:x2, y1:y2]
            label_strong = np.copy(label_weak)

            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                data_weak, label_weak = self.flip(data_weak, label_weak)
                data_strong, label_strong = self.flip(data_strong, label_strong)
            if np.random.rand() < 0.7:
                data_strong = self.radiation_noise(data_strong)
            if np.random.rand() < 0.7:
                data_strong = self.mixture_noise(data_strong, label_strong)
            if np.random.rand() < 0.7:
                data_strong = self.pca_augmentation(data_strong, label_strong, M=1.1)

            # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
            data_weak = np.asarray(np.copy(data_weak).transpose((2, 0, 1)), dtype='float32')

            data_strong = np.asarray(np.copy(data_strong).transpose((2, 0, 1)), dtype='float32')

            # Load the data into PyTorch tensors
            data_weak = torch.from_numpy(data_weak)
            data_strong = torch.from_numpy(data_strong)

            # Add a fourth dimension for 3D CNN
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                data_weak = data_weak.unsqueeze(0)
                data_strong = data_strong.unsqueeze(0)
            return data_weak, data_strong

def get_pixel_idx(data, gt, ignored_labels, patch_size):
    """
    Args:
        data: list of 3D HSI patches
        gt: list of 2D array of labels
    Output:
        idx_sup: list of indexes for supervised data
        idx_val: list of indexes for validation data
        idx_unsup: list of indexes for unsupervised data
    """
    mask = np.ones_like(gt)
    for l in ignored_labels:
        mask[gt == l] = 0
    patch_labeled, x_labeled, y_labeled = np.nonzero(mask)

    for l in ignored_labels:
        patch_unlabeled, x_unlabeled, y_unlabeled = np.nonzero(mask==l)

    p = patch_size // 2
    x_patch_size = data[0].shape[0]
    y_patch_size = data[0].shape[1]

    idx_labeled = np.array([(p_l, x_l, y_l) for p_l, x_l, y_l in zip(patch_labeled, x_labeled, y_labeled) if x_l >= p and x_l < x_patch_size - p and y_l >= p and y_l < y_patch_size - p])
    np.random.shuffle(idx_labeled)

    ratio = int(0.95*len(idx_labeled))

    idx_sup = idx_labeled[:ratio]
    idx_val = idx_labeled[ratio:]

    idx_unsup = np.array([(p_u, x_u, y_u) for p_u, x_u, y_u in zip(patch_unlabeled, x_unlabeled, y_unlabeled) if x_u >= p and x_u < x_patch_size - p and y_u >= p and y_u < y_patch_size - p])

    return idx_sup, idx_val, idx_unsup


class HyperX_patches(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene with list of 3D HSI """

    def __init__(self, data, gt, idx, labeled=True, **args):
        """
        Args:
            data: list of 3D hyperspectral image patches
            gt: list of 2D array of labels
            idx: list of indices from where to pull samples
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX_patches, self).__init__()
        self.data = np.array(data)
        self.label = np.array(gt)
        self.idx = idx
        self.patch_size = args['patch_size']
        self.name = args['dataset']
        self.ignored_labels = set(args['ignored_labels'])
        #Augmentations
        self.flip_augmentation = args['flip_augmentation']
        self.radiation_augmentation = args['radiation_augmentation']
        self.mixture_augmentation = args['mixture_augmentation']
        self.pca_aug = args['pca_augmentation']
        self.pca_strength = args['pca_strength']
        self.spatial_cutout_aug = args['cutout_spatial']
        self.spectral_cutout_aug = args['cutout_spectral']
        self.M = args['augmentation_magnitude']
        self.spatial_comb = args['spatial_combinations']
        self.mean_spectral = args['spectral_mean']
        self.spectral_mvavg = args['moving_average']

        self.center_pixel = args['center_pixel']
        self.labeled = labeled



        """
        mask = np.ones_like(self.label)
        for l in self.ignored_labels:
            mask[self.label == l] = 0
        patch_labeled, x_labeled, y_labeled = np.nonzero(mask)

        for l in self.ignored_labels:
            patch_unlabeled, x_unlabeled, y_unlabeled = np.nonzero(mask==l)

        p = self.patch_size // 2
        self.indices_labled = np.array([(p_l, x_l, y_l) for p_l, x_l, y_l in zip(patch_labeled, x_labeled, y_labeled) if x_l >= p and x_l < data.shape[0] - p and y_l >= p and y_l < data.shape[1] - p])
        self.labels = [self.label[p_l, x_l, y_l] for p_l, x_l, y_l in self.indices_labeled]

        self.indices_unlabled = np.array([(p_u, x_u, y_u) for p_u, x_u, y_u in zip(patch_unlabeled, x_unlabeled, y_unlabeled) if x_u >= p and x_u < data.shape[0] - p and y_u >= p and y_u < data.shape[1] - p])
        """
        if self.labeled == True:
            self.labels = [self.label[p_l, x_l, y_l] for p_l, x_l, y_l in self.idx]

        self.idx_shuffle = np.copy(self.idx)
        np.random.shuffle(self.idx_shuffle)

        """
        self.class_var = {}
        for c in np.unique(self.label):
            if c not in self.ignored_labels:
                l_indices = np.nonzero(self.labels==c)
                pos = self.indices_labeled[l_indices]
                var = np.var(self.data[pos[:,0], pos[:,1]], axis=0)
                self.class_var[c] = np.diag(var)
        """

        self.data_mean = np.mean(self.data, axis=(0,1,2))

        centered_data = self.data - self.data_mean
        data_train = np.array([centered_data[p_l, x_l, y_l] for p_l, x_l, y_l in self.idx])
        self.pca = PCA(n_components=11)
        self.pca.fit(data_train)


    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                #This is the original implementation, but I think it takes the wrong data
                #x, y = self.indices_shuffle[l_indice]
                #This is the new implementaiton, it does not mix indices and should take the right sample
                p, x, y = self.idx[l_indice]
                data2[idx] = self.data[p,x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    #PCA augmentation technique. Adds noise in pca space and transform back
    def pca_augmentation(self, data, M=1):
        data_aug = np.zeros_like(data)
        data_train = data - self.data_mean
        for idx, _ in np.ndenumerate(data[:,:,0]):
            x,y = idx
            alpha = M*np.random.uniform(0.5,1.5)
            data_pca = self.pca.transform(data_train[x,y,:].reshape(1,-1))
            data_pca[:,0] = data_pca[:,0]*alpha
            #alpha = M*np.random.uniform(0.5,1.5,11)
            #data_pca = data_pca*alpha
            data_aug[x,y,:] = self.pca.inverse_transform(data_pca)
        data_aug = data_aug + self.data_mean
        return data_aug

    #Cutout augmentation, cut out a random part of the image and replace with ignored label
    #this is the vanilla implementation, might not work for this case because
    #of the middle pixel. The middle pixel should not be cut out since it is everything
    #[0 0 0 0 0]
    #[0 0 0 0 0]
    #[0 0 P 0 0]
    #[0 0 0 0 0]
    #[0 0 0 0 0]
    #fucked up augmentation purely made for 5x5 patches...
    """
    def cutout_spatial(self, image):
        cutout_image = image
        y = np.random.choice([-1,0,1])
        if y == 0:
            x = np.random.choice([-1,1])
            x_step = 2*x
            x1 = np.min([x, x_step]) + 2
            x2 = np.max([x, x_step]) + 2

            y_step = np.random.choice([-1,1])
            y1 = np.min([y, y_step]) + 2
            y2 = np.max([y, y_step]) + 2
        else:
            x = np.random.choice([-1,0,1])
            if x == 0:
                x_step = np.random.choice([-1,1])
                x1 = np.min([x, x_step]) + 2
                x2 = np.max([x, x_step]) + 2
            else:
                x_step = np.random.choice([-1,1])
                x1 = np.min([x, x_step]) + 2
                x2 = np.max([x, x_step]) + 2
            y_step = 2*y
            y1 = np.min([y, y_step]) + 2
            y2 = np.max([y, y_step]) + 2
        cutout_image[y1:y2,x1:x2,:] = 0
        return cutout_image
    """
    def cutout_spatial(self, data):
        cutout_image = np.copy(data)
        x,y = data.shape[:2]
        x_c, y_c = np.random.randint(x, size=2)
        if x_c == x//2 and y_c == y//2:
            return cutout_image
        else:
            cutout_image[x_c, y_c, :] = 0
            return cutout_image
    #Hyperspectral cutout method to cutout part of the spectral bands
    def cutout_spectral(self, image, M=1):
        h, w, channels = image.shape
        #See if magnitude works as factor
        cutouts = 1*M
        bands = 2*M
        new_image = image
        for _ in range(cutouts):
            c = np.random.randint(channels)
            c1 = np.clip(c - bands // 2, 0, channels)
            c2 = np.clip(c1 + bands, 0, channels)
            new_image[:,:,c1:c2] = 0
        return new_image

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

    def identity(data):
        return data

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):

        """
        If the dataset is labeled return data accoring with data augmentation
        and the ground truth of that specific data.
        """
        if self.labeled == True:
            p, x, y = self.idx_shuffle[i]
            x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size

            data = self.data[p, x1:x2, y1:y2]
            label = self.label[p, x1:x2, y1:y2]

            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                data, label = self.flip(data, label)
            if self.radiation_augmentation and np.random.random() < 0.5:
                data = self.radiation_noise(data)
            if self.mixture_augmentation and np.random.random() < 0.5:
                data = self.mixture_noise(data, label)
            if self.pca_aug and np.random.random() < 0.5:
                data = self.pca_augmentation(data, M=self.pca_strength)
            if self.spatial_cutout_aug and np.random.random() < 0.5:
                data = self.cutout_spatial(data)
            if self.spatial_comb and np.random.random() < 0.5:
                data = self.spatial_combinations(data, self.M)
            if self.mean_spectral and np.random.random() < 0.5:
                data = self.spectral_mean(data, self.M)
            if self.spectral_mvavg and np.random.random() < 0.5:
                data = self.moving_average(data, self.M)


            # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
            data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
            label = np.asarray(np.copy(label), dtype='int64')

            # Load the data into PyTorch tensors
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            # Extract the center label if needed
            if self.center_pixel and self.patch_size > 1:
                label = label[self.patch_size // 2, self.patch_size // 2]
            # Remove unused dimensions when we work with invidual spectrums
            elif self.patch_size == 1:
                data = data[:, 0, 0]
                label = label[0, 0]

            # Add a fourth dimension for 3D CNN
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                data = data.unsqueeze(0)
            return data, label

        """
        If the dataset is labeled return data accoring with data augmentation
        and the ground truth of that specific data.
        """
        if self.labeled == 'Val':
            p, x, y = self.idx_shuffle[i]
            x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size

            data = self.data[p, x1:x2, y1:y2]
            label = self.label[p, x1:x2, y1:y2]

            # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
            data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
            label = np.asarray(np.copy(label), dtype='int64')

            # Load the data into PyTorch tensors
            data = torch.from_numpy(data)
            label = torch.from_numpy(label)
            # Extract the center label if needed
            if self.center_pixel and self.patch_size > 1:
                label = label[self.patch_size // 2, self.patch_size // 2]
            # Remove unused dimensions when we work with invidual spectrums
            elif self.patch_size == 1:
                data = data[:, 0, 0]
                label = label[0, 0]

            # Add a fourth dimension for 3D CNN
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                data = data.unsqueeze(0)
            return data, label

        """
        If the data is unlabeled, return one version of the data with weak
        data augmentation and one version of the same data with strong
        data augmentation.
        """
        if self.labeled == False:
            p, x, y = self.idx_shuffle[i]
            x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size

            data_weak = self.data[p, x1:x2, y1:y2]
            data_strong = np.copy(data_weak)
            #print(data_strong.shape)
            #This will currently not work with data that has no labels, it doesnt harm it right now either though...
            # i.e. it is a zero matrix
            #label_weak = self.label[p, x1:x2, y1:y2]
            #label_strong = np.copy(label_weak)

            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                #data_weak, label_weak = self.flip(data_weak, label_weak)
                #data_strong, label_strong = self.flip(data_strong, label_strong)
                data_weak = self.flip(data_weak)
                data_strong = self.flip(data_strong)
            if np.random.rand() < 0.5:
                data_strong = self.radiation_noise(data_strong)
            #if np.random.rand() < 0.7:
                #data_strong = self.mixture_noise(data_strong, label_strong)
            if np.random.rand() < 0.5:
                data_strong = self.pca_augmentation(data_strong, M=self.pca_strength)
            if np.random.random() < 0.5 and self.patch_size > 1:
                data_strong = self.spatial_combinations(data_strong, self.M)
            if np.random.random() < 0.5:
                data_strong = self.spectral_mean(data_strong, self.M)
            if np.random.random() < 0.5:
                data_strong = self.moving_average(data_strong, self.M)

            data_strong = self.cutout_spatial(data_strong)

            # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
            data_weak = np.asarray(np.copy(data_weak).transpose((2, 0, 1)), dtype='float32')

            data_strong = np.asarray(np.copy(data_strong).transpose((2, 0, 1)), dtype='float32')

            # Load the data into PyTorch tensors
            data_weak = torch.from_numpy(data_weak)
            data_strong = torch.from_numpy(data_strong)

            if self.patch_size == 1:
                data_weak = data_weak[:, 0, 0]
                data_strong = data_strong[:, 0, 0]

            # Add a fourth dimension for 3D CNN
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                data_weak = data_weak.unsqueeze(0)
                data_strong = data_strong.unsqueeze(0)
            return data_weak, data_strong

        """
        Version of fetching a sample when using Mean Teacher Model
        """
        if self.labeled == 'Mean':
            p, x, y = self.idx_shuffle[i]
            x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size

            data_weak = self.data[p, x1:x2, y1:y2]
            data_strong = np.copy(data_weak)
            #print(data_strong.shape)
            #This will currently not work with data that has no labels, it doesnt harm it right now either though...
            # i.e. it is a zero matrix
            #label_weak = self.label[p, x1:x2, y1:y2]
            #label_strong = np.copy(label_weak)

            if self.flip_augmentation and self.patch_size > 1:
                # Perform data augmentation (only on 2D patches)
                #data_weak, label_weak = self.flip(data_weak, label_weak)
                #data_strong, label_strong = self.flip(data_strong, label_strong)
                data_weak = self.flip(data_weak)
                data_strong = self.flip(data_strong)
            if np.random.rand() < 0.5:
                data_strong = self.radiation_noise(data_strong)
                data_weak = self.radiation_noise(data_weak)
            #if np.random.rand() < 0.7:
                #data_strong = self.mixture_noise(data_strong, label_strong)
            if np.random.rand() < 0.5:
                data_strong = self.pca_augmentation(data_strong, M=self.pca_strength)
                data_weak = self.pca_augmentation(data_weak, M=self.pca_strength)
            if np.random.random() < 0.5 and self.patch_size > 1:
                data_strong = self.spatial_combinations(data_strong, self.M)
                data_weak = self.spatial_combinations(data_weak, self.M)
            if np.random.random() < 0.5:
                data_strong = self.spectral_mean(data_strong, self.M)
                data_weak = self.spectral_mean(data_weak, self.M)
            if np.random.random() < 0.5:
                data_strong = self.moving_average(data_strong, self.M)
                data_weak = self.moving_average(data_weak, self.M)

            data_strong = self.cutout_spatial(data_strong)
            data_weak = self.cutout_spatial(data_weak)

            # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
            data_weak = np.asarray(np.copy(data_weak).transpose((2, 0, 1)), dtype='float32')

            data_strong = np.asarray(np.copy(data_strong).transpose((2, 0, 1)), dtype='float32')

            # Load the data into PyTorch tensors
            data_weak = torch.from_numpy(data_weak)
            data_strong = torch.from_numpy(data_strong)

            if self.patch_size == 1:
                data_weak = data_weak[:, 0, 0]
                data_strong = data_strong[:, 0, 0]

            # Add a fourth dimension for 3D CNN
            if self.patch_size > 1:
                # Make 4D data ((Batch x) Planes x Channels x Width x Height)
                data_weak = data_weak.unsqueeze(0)
                data_strong = data_strong.unsqueeze(0)
            return data_weak, data_strong
