import os
import argparse
from fixmatch_hsi import main as fixmatch
from supervised_hsi import main as supervised
from mixup_hsi import main as mixup
import numpy as np


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Main testing file for running several tests over several datasets')
    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='Name of dataset to run. Salinas, PaviaU or Indian, defaults to Salinas')
    parser.add_argument('--runs', type=int, default=1,
                        help='Amount of time to run on each dataset/fold. Defaults to 1.')
    parser.add_argument('--data_sampling', type=str, default='nalepa',
                        help='What kind of sampling of data to run. random, disjoint, region or patch based. Defaults to patch (nalepa).')
    parser.add_argument('--run_name', type=str, default='new',
                        help='Folder name to save all the results to. Defaults to results/new/')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Amount of epochs. Defaults to 20')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Defaults to 0.001')
    parser.add_argument('--method', type=str, default='supervised',
                        help='Which method to use, supervised, mixup or fixmatch. Defaults to supervised.')
    parser.add_argument('--server', action='store_true',
                        help='Use to run on server and sample from the designated folder.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size. Defaults to 10.')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence threshold for fixmatch method. Defaults to 0.95.')

    args = parser.parse_args(raw_args)

    results = []

    if args.server:
        data_path = '/data/ieee_supplement/Hyperspectral_Grids/{}'
    else:
        data_path = '/home/oscar/Desktop/Exjobb/Data/ieee_supplement/Hyperspectral_Grids/{}'

    if args.dataset == 'Indian':
        folds = 4
    else:
        folds = 5

    avg_acc = np.zeros(folds)

    for f in range(0,folds):
        for r in range(args.runs):
            print('Running: ' + str(r) + 'time and: ' + str(f) + ' fold.')
            if args.method == 'supervised':
                result = supervised(['--data_dir', data_path.format(args.dataset), '--results', 'results/{}/'.format(args.run_name),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--flip_augmentation'])
            elif args.method == 'mixup':
                result = mixup(['--data_dir', data_path.format(args.dataset), '--results', 'results/{}/'.format(args.run_name),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--flip_augmentation'])
            elif args.method == 'fixmatch':
                result = fixmatch(['--class_balancing', '--data_dir', data_path.format(args.dataset), '--results', 'results/{}/'.format(args.run_name),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', '--threshold', '{}'.format(args.threshold)])
            else:
                print('No method with this name')
                results = None
            results.append(result)
            avg_acc[f] += result['Accuracy']
            print(avg_acc[f])

    avg_acc = avg_acc/args.runs
    print('Ran all the folds.')
    """
    for f in range(0,folds):
        average_acc = results[f*args.runs:f*args.runs+args.runs]['Accuracy']
        avg_acc[f] = np.sum(average_acc)/args.runs
    """
    print('Average accuracy per fold: ' + str(avg_acc))

    print('Total average accuracy: ' + str(np.sum(avg_acc)/len(avg_acc)))

if __name__ == '__main__':
    thresholds = [0.7, 0.8, 0.9, 0.95]
    for t in thresholds:
        main(['--threshold', str(t), '--runs', str(1), '--epochs', str(1), '--method', 'fixmatch'])
