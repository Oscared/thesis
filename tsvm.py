from qns3vm import QN_S3VM
import scipy
import numpy as np
import os
import random


import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 

dataset_path = '/home/oscar/Desktop/Exjobb/Data/ieee_supplement/Hyperspectral_Grids/Salinas/'

def main():
    data_path = '/home/oscar/Desktop/Exjobb/Data/ieee_supplement/Hyperspectral_Grids/Salinas/'

    fold = 0

    train_img, train_gt, test_img, test_gt, label_values, ignored_labels, rgb_bands, palette = get_patch_data('Salinas', 1, target_folder=data_path, fold=fold)

    def convert_to_color(x):
        return convert_to_color_(x, palette=palette)
    def convert_from_color(x):
        return convert_from_color_(x, palette=invert_palette)

    n_bands = train_img.shape[-1]
    n_classes = len(label_values) - len(ignored_labels)

    idx_sup, idx_val, idx_unsup = get_pixel_idx(train_img, train_gt, ignored_labels, 5)

    i = 0
    X = np.zeros((len(idx_sup), n_bands))
    Y = np.zeros(len(idx_sup))
    for p, x, y in idx_sup:
        X[i, :] = train_img[p,x,y]
        Y[i] = train_gt[p,x,y] - 1
        i += 1

    i = 0
    X_un = np.zeros((len(idx_unsup), n_bands))
    for p, x, y in idx_unsup:
        X_un[i, :] = train_img[p,x,y]
        i += 1

    G = 10

    C_labeled = 1.0
    C_unlabeled = np.zeros(G)
    C_unlabeled[0] = C_labeled/(10*G)
    C_max = 10.0

    sigma = 1.0
    Np = 0
    Nm = 0
    A = np.zeros(n_classes)

    rand_generator = random.Random()

    CLF = []

    for i in range(n_classes):
        idx_class = Y == i
        idx_rest = Y != i
        X_train = np.concatenate((X[idx_class], X[idx_rest]))
        Y_train = np.concatenate((np.ones(len(X[idx_class])).astype(int), -np.ones(len(X[idx_rest])).astype(int)))
        CLF.append(QN_S3VM(X_train.tolist(), Y_train.tolist(), X_un.tolist(), rand_generator, lam=C_labeled, lamU=C_unlabeled[0], kernel_type='RBF'))
        CLF[i].train()

        for x in X_train:
            value = CLF[i].predictValue(x.tolist())
            if value == 1.0:
                Np += 1
            if value == -1.0:
                Nm += 1
        A[i] = np.min(Np,Nm)

    for i in range(G):
        for i in range(n_classes):
            #Find A transductive samples
            values = []
            for x in X_un:
                values.append(CLF[i].predictValue(x))
            values = np.asarray(values)
            values_p = np.abs(1 - values[values>0])
            values_m = np.abs(-1 - values[values<0])

            idx_p = np.zeros(A)
            idx_m = np.zeros(A)

            for i in range(A):
                idx_p[i] = np.argmin(values_p)
                idx_m[i] = np.argmin(values_m)
                values_p[idx_p[i]] = 100000
                values_m[idx_m[i]] = 100000

            #Threshold
            D_p = np.sum(values[idx_p])/A
            D_m = np.sum(values[idx_m])/A

            Th_p = D_p*np.max(np.abs(values[idx_p]))
            Th_m = D_m*np.max(np.abs(values[idx_m]))

            #Trim candidate set
            N_p = len(values[idx_p][np.abs(values[idx_p])>=Th_p])
            N_m = len(values[idx_m][np.abs(values[idx_m])>=Th_m])
            N = np.min(N_p, N_m)

            if N == N_p:
                X_un_p = X_un[idx_p]
                X_un_m = X_un[idx_m][values_m.argsort()[-N:][::-1]]
                del_idx = np.concatenate((idx_p, idx_m[[values_m.argsort()[-N:][::-1]]]))
            elif N==N_m:
                X_un_m = X_un[idx_m]
                X_un_p = X_un[idx_p][values_p.argsort()[-N:][::-1]]
                del_idx = np.concatenate((idx_m, idx_p[[values_p.argsort()[-N:][::-1]]]))

            #Update datasets
            idx_class = Y == i
            idx_rest = Y != i
            X_train = np.stack(X[idx_class], X[idx_rest])
            Y_train = np.stack(np.ones(len(X[idx_class])), -np.ones(len(X[idx_rest])))

            X_train.append(X_un_m, X_un_p)
            Y_train.append(-np.ones(len(X_un_m)), np.ones(len(X_un_p)))
            np.delete(X_un, del_idx)

            #Update weight factor
            C[i] = (C_max - C[0])/G^2*i^2 + C[0]

            #Retrain the TSVM
            CLF[i] = QN_S3VM(X_train, Y_train, X_un, lam=C_labeled, lamU=C[i], kernel_type='RBF')


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.
    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)
    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.
    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)
    Returns:
        arr_2d: int 2D array of labels
    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

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

    if dataset_name == 'salinas':
        remove_files = 3
    elif dataset_name == 'pavia':
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

    train_img = train_patches
    train_gt = np.asarray(train_patches_gt, dtype='int')
    test_patch = test_img
    test_gt = np.asarray(test_img_gt, dtype='int')

    train_img = np.asarray(train_img, dtype='float32')
    test_patch = np.asarray(test_patch, dtype='float32')


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
    else:
        print("Error: no dataset of the requested type found. Available datasets are PaviaU, Salinas.")

    return train_img, train_gt, test_patch, test_gt, label_values, ignored_labels, rgb_bands, palette


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


if __name__ == '__main__':
    main()
