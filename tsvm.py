import scipy
import seaborn as sns
import numpy as np
import os
import random
import sklearn.svm as SVM
import argparse
import visdom
import utils

from datasets import get_dataset

dataset_path = '/data/ieee_supplement/Hyperspectral_Grids/Salinas/'

def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Transductive SVM implementation.')
    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='Name of dataset to run. Salinas, Pavia or Indian, defaults to Salinas')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold to sample from if using Nalepas validation scheme')
    parser.add_argument('--use_vis', action='store_true',
                        help='use to enable Visdom for visualization, remember to start the Visdom server')
    parser.add_argument('--samples_per_class', type=int, default=10,
                        help='Amount of samples to sample for each class when sampling a fixed amount. Defaults to 10.')
    parser.add_argument('--sampling_fixed', type=str, default='False',
                        help='Use to sample a fixed amount of samples for each class from Nalepa sampling')
    parser.add_argument('--extra_data', type=str, default='True',
                        help='add extra data for pavia dataset. Defaults to true.')

    args = parser.parse_args(raw_args)

    if args.use_vis == True:
        vis = visdom.Visdom()
    else:
        vis = None

    data_path = '/data/ieee_supplement/Hyperspectral_Grids/{}'

    if args.dataset == 'Salinas':
        data_folder = 'Salinas'
    elif args.dataset == 'Pavia':
        data_folder = 'Pavia University'
    elif args.dataset == 'Indian':
        data_folder = 'Indian Pines'
    else:
        print('No dataset by right name')

    train_img, train_gt, test_img, test_gt, label_values, ignored_labels, rgb_bands, palette = get_patch_data(args.dataset, 1, target_folder=data_path.format(data_folder), fold=args.fold)
    if args.dataset == 'Pavia' and args.extra_data == 'True':
        train_img, train_gt, test_img, test_gt, label_values, ignored_labels, rgb_bands, palette = get_patch_data('pavia', 1, target_folder=data_path.format(data_folder), fold=args.fold)
        args.n_bands = train_img.shape[-1]
        img_unlabeled, _, _, _, _, _ = get_dataset('PaviaC', target_folder='/data/')

        img_unlabeled = np.concatenate((img_unlabeled, img_unlabeled[:,:,-1, np.newaxis]), axis=-1)

    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", len(label_values) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    invert_palette = {v: k for k, v in palette.items()}

    def convert_to_color(x):
        return convert_to_color_(x, palette=palette)
    def convert_from_color(x):
        return convert_from_color_(x, palette=invert_palette)

    n_bands = train_img.shape[-1]
    n_classes = len(label_values) - len(ignored_labels)

    idx_sup, idx_val, idx_unsup = get_pixel_idx(train_img, train_gt, ignored_labels, 5)

    if args.sampling_fixed == 'True':
        unique_labels = np.zeros(len(label_values))
        new_idx_sup = []
        index = 0
        for p,x,y in idx_sup:
            label = train_gt[p,x,y]
            if unique_labels[label] < args.samples_per_class:
                unique_labels[label] += 1
                new_idx_sup.append([p,x,y])
                np.delete(idx_sup, index)
            index += 1
        idx_unsup = np.concatenate((idx_sup, idx_unsup))
        idx_sup = np.asarray(new_idx_sup)

    i = 0
    X = np.zeros((len(idx_sup), n_bands))
    Y = np.zeros(len(idx_sup))
    for p, x, y in idx_sup:
        X[i, :] = train_img[p,x,y]
        Y[i] = train_gt[p,x,y] - 1
        i += 1

    i = 0
    if args.dataset == 'Pavia' and args.extra_data == 'True':
        X_un = np.zeros((len(idx_unsup) + img_unlabeled.shape[0]*img_unlabeled.shape[1], n_bands))
    else:
        X_un = np.zeros((len(idx_unsup), n_bands))
    for p, x, y in idx_unsup:
        X_un[i, :] = train_img[p,x,y]
        i += 1
    if args.dataset == 'Pavia' and args.extra_data == 'True':
        for x in range(img_unlabeled.shape[0]):
            for y in range(img_unlabeled.shape[1]):
                X_un[i,:] = img_unlabeled[x,y,:]
                i += 1

    print('Starting TSVM...')

    G = 2

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

    no_go = []

    for i in range(n_classes):
        idx_class = Y == i
        idx_rest = Y != i
        X_train = np.concatenate((X[idx_class], X[idx_rest]))
        Y_train = np.concatenate((np.ones(len(X[idx_class])).astype(int), -np.ones(len(X[idx_rest])).astype(int)))
        # Classifier is a transductive SVM
        #CLF.append(QN_S3VM(X_train.tolist(), Y_train.tolist(), X_un.tolist(), rand_generator, lam=C_labeled, lamU=C_unlabeled[0], kernel_type='RBF'))
        #CLF[i].train()

        # Classifier is a standard SVM
        CLF.append(SVM.SVC(kernel='rbf', gamma=0.5, C=C_labeled)) #C=1000 other option
        if np.max(idx_class) == 0:
            no_go.append(i)
        else:
            CLF[i].fit(X_train,Y_train)

            support = CLF[i].n_support_
            Np = support[1]
            Nm = support[0]

            A[i] = np.min((Np,Nm))

    print('No go classes: ' + str(no_go))
    yes_go = np.delete(range(n_classes), no_go)

    for i in yes_go:
        X_un_run = X_un
        for g in range(G):
            #print('Running class: ' + str(i) + '. Time: ' + str(g))
            #Find A transductive samples
            values = CLF[i].decision_function(X_un)
            values = np.asarray(values)

            if np.max(values)>0 and np.min(values)<0:
                #Extract values of positive unlabeled samples
                values_p = np.abs(1 - values[values>0])
                idx_p = values_p.argsort()[:int(A[i])]
                #Threshold
                D_p = np.sum(values[idx_p])/A[i]
                Th_p = D_p*np.max(np.abs(values[idx_p]))
                #Trim candidate set
                N_p = len(values[idx_p][np.abs(values[idx_p])>=Th_p])


                values_m = np.abs(-1 - values[values<0])
                idx_m = values_m.argsort()[:int(A[i])]
                #Threshold
                D_m = np.sum(values[idx_m])/A[i]
                Th_m = D_m*np.max(np.abs(values[idx_m]))
                #Trim candidate set
                N_m = len(values[idx_m][np.abs(values[idx_m])>=Th_m])

                N = np.min((N_p, N_m))

                if N == N_p:
                    X_un_p = X_un_run[idx_p]
                    X_un_m = X_un_run[idx_m[:N]]
                    del_idx = np.concatenate((idx_p, idx_m[:N]))
                elif N==N_m:
                    X_un_m = X_un_run[idx_m]
                    X_un_p = X_un_run[idx_p[:N]]
                    del_idx = np.concatenate((idx_m, idx_p[:N]))

                #Update datasets
                idx_class = Y == i
                idx_rest = Y != i
                X_train = np.concatenate((X[idx_class], X[idx_rest]))
                Y_train = np.concatenate((np.ones(len(X[idx_class])), -np.ones(len(X[idx_rest]))))

                X_train = np.append(X_train, np.concatenate((X_un_m, X_un_p)), axis=0)
                Y_train = np.append(Y_train, np.concatenate((-np.ones(len(X_un_m)), np.ones(len(X_un_p)))), axis=0)
                X_un_run = np.delete(X_un_run, del_idx,0)

                #Update weight factor
                #C[i] = (C_max - C[0])/G^2*g^2 + C[0]

                #Retrain the TSVM
                CLF[i].fit(X_train, Y_train)

    pred_values = np.zeros((n_classes, test_img.shape[0]*test_img.shape[1]))
    for i in yes_go:
        pred_values[i,:] = (CLF[i].decision_function(test_img.reshape(-1, n_bands)))
    predicted_values = np.asarray(pred_values)
    prediction = np.argmax(predicted_values, axis=0)
    prediction = prediction.reshape(test_img.shape[:2])

    run_results = utils.metrics(prediction, test_gt, ignored_labels=ignored_labels, n_classes=n_classes)

    mask = np.zeros(test_gt.shape, dtype='bool')
    for l in ignored_labels:
        mask[test_gt == l] = True
    prediction += 1
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    utils.display_predictions(color_prediction, vis, gt=convert_to_color(test_gt), caption="Prediction vs. test ground truth")

    utils.show_results(run_results, vis, label_values=label_values)

    return run_results

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
    datasets = ['Pavia', 'Salinas', 'Indian']
    runs = 2
    results = []
    for dataset in datasets:
        if dataset == 'Indian':
            folds = 4
        else:
            folds = 5

        avg_acc = np.zeros(folds)

        for f in range(0,folds):
            for r in range(runs):
                result = main(['--dataset', dataset, '--fold', str(f)])
                results.append(result)
                avg_acc[f] += result['Accuracy']

        avg_acc = avg_acc/args.runs
        print('Ran all the folds for: ' + dataset)
        print('Average accuracy per fold: ' + str(avg_acc))
        print('Total average accuracy: ' + str(np.sum(avg_acc)/len(avg_acc)))
        print(results)
