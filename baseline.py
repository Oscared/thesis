import visdom
from datasets import get_dataset, get_patch_data, get_pixel_idx, HyperX, HyperX_patches
import utils
import numpy as np
import seaborn as sns

import torch
from tensorboardX import SummaryWriter
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn
import sklearn.svm as SVM

import math
import os
import datetime
#import joblib
from tqdm import tqdm
import argparse

def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Hyperspectral image classification with FixMatch")
    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='Name of dataset to run, Salinas or PaviaU')
    parser.add_argument('--save', action='store_true',
                        help='use to save model weights when running')

    parser.add_argument('--results', type=str, default='results',
                        help='where to save results to (default results)')
    parser.add_argument('--save_dir', type=str, default='/saves/',
                        help='where to save models to (default /saves/)')
    parser.add_argument('--data_dir', type=str, default='/data/',
                        help='where to fetch data from (default /data/)')
    parser.add_argument('--load_file', type=str, default=None,
                        help='wihch file to load weights from (default None)')
    parser.add_argument('--use_vis', action='store_true',
                        help='use to enable Visdom for visualization, remember to start the Visdom server')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold to sample from if using Nalepas validation scheme')
    parser.add_argument('--sampling_fixed', type=str, default='True',
                        help='Use to sample a fixed amount of samples for each class from Nalepa sampling')
    parser.add_argument('--samples_per_class', type=int, default=10,
                        help='Amount of samples to sample for each class when sampling a fixed amount. Defaults to 10.')


    parser.add_argument('--supervision', type=str, default='full',
                        help='check this more, use to make us of all labeled or not, full or semi')

    args = parser.parse_args(raw_args)

    device = utils.get_device(args.cuda)
    args.device = device

    if args.use_vis == True:
        vis = visdom.Visdom()
    else:
        vis = None

    tensorboard_dir = str(args.results + '/' + datetime.datetime.now().strftime("%m-%d-%X"))

    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    train_img, train_gt, test_img, test_gt, label_values, ignored_labels, rgb_bands, palette = get_patch_data(args.dataset, args.patch_size, target_folder=args.data_dir, fold=args.fold)
    args.n_bands = train_img.shape[-1]

    args.n_classes = len(label_values) - len(ignored_labels)
    args.ignored_labels = ignored_labels

    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", len(label_values) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    invert_palette = {v: k for k, v in palette.items()}

    def convert_to_color(x):
        return utils.convert_to_color_(x, palette=palette)
    def convert_from_color(x):
        return utils.convert_from_color_(x, palette=invert_palette)

    print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(train_gt) + np.count_nonzero(test_gt)))
    writer.add_text('Amount of training samples', "{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(test_gt)))

    utils.display_predictions(convert_to_color(test_gt), vis, writer=writer,
                                  caption="Test ground truth")


    #Get fixed amount of random samples for validation
    idx_sup, idx_val, idx_unsup = get_pixel_idx(train_img, train_gt, args.ignored_labels, args.patch_size)

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

    writer.add_text('Amount of labeled training samples', "{} samples selected (over {})".format(idx_sup.shape[0], np.count_nonzero(train_gt)))
    train_labeled_gt = [train_gt[p_l, x_l, y_l] for p_l, x_l, y_l in idx_sup]

    samples_class = np.zeros(args.n_classes)
    for c in np.unique(train_labeled_gt):
        samples_class[c-1] = np.count_nonzero(train_labeled_gt == c)
    writer.add_text('Labeled samples per class', str(samples_class))
    print('Labeled samples per class: ' + str(samples_class))

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

    amount_labeled = idx_sup.shape[0]

    print(args)
    writer.add_text('Arguments', str(args))

    if args.model == 'SVM':
        clf = SVM.SVC(kernel='rbf', gamma=0.5, C=1000)
        clf.fit(X, Y)
        prediction = clf.predict(test_img.reshape(-1, n_bands))
        prediction = prediction.reshape(test_img.shape[:2])
    elif args.model == 'RF':
        clf = RandomForestClassifier(n_estimators = 200, min_samples_split=2, \
                    max_features=10, max_depth=10)
        clf.fit(X, Y)
        prediction = clf.predict(test_img.reshape(-1, n_bands))
        prediction = prediction.reshape(test_img.shape[:2])
    elif args.model == 'XGBOOST':
        clf = XGBClassifier()
        clf.fit(X, Y)
        prediction = clf.predict(test_img.reshape(-1, n_bands))
        prediction = prediction.reshape(test_img.shape[:2])


    run_results = utils.metrics(prediction, test_gt,
                                ignored_labels=args.ignored_labels,
                                n_classes=args.n_classes)

    mask = np.zeros(test_gt.shape, dtype='bool')
    for l in args.ignored_labels:
        mask[test_gt == l] = True
    prediction += 1
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    utils.display_predictions(color_prediction, vis, gt=convert_to_color(test_gt), writer=writer,
                              caption="Prediction vs. test ground truth")

    utils.show_results(run_results, vis, writer=writer, label_values=label_values)

    writer.close()

    return run_results