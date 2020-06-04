import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.decomposition import PCA

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
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
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


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, labeled=True, pca_aug=False, **args):
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
        self.pca_aug = pca_aug
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
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        #np.random.shuffle(self.indices)

        self.class_var = {}
        for c in np.unique(self.label):
            if c not in self.ignored_labels:
                l_indices = np.nonzero(self.labels==c)
                pos = self.indices[l_indices]
                var = np.var(self.data[pos[:,0], pos[:,1]], axis=0)
                self.class_var[c] = np.diag(var)

        if self.pca_aug:
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
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def pca_augmentation(self, data, strength=1):
        data_train = data - np.mean(data)
        alpha = strength*np.random.uniform(0.9,1.1, np.shape(data)[0])
        data_pca = self.pca.transform(data_train)
        data_pca[:,0] = data_pca[:,0]*alpha
        data_aug = self.pca.inverse_transform(data_pca)
        return data_aug

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
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
                data_strong = self.pca_augmentation(data_strong, strength=1.1)

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
