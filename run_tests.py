import os
import argparse
from fixmatch_hsi import main as fixmatch
from supervised_hsi import main as supervised
from mixup_hsi import main as mixup
import numpy as np
from tensorboardX import SummaryWriter


def main(raw_args=None):
    parser = argparse.ArgumentParser(description='Main testing file for running several tests over several datasets')
    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='Name of dataset to run. Salinas, Pavia or Indian, defaults to Salinas')
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
    parser.add_argument('--sampling_fixed', type=str, default='False',
                        help='Use to sample a fixed amount of samples per class for nalepa sampling.')
    parser.add_argument('--model', type=str, default='3D',
                        help='Model to use 3D or 1D CNN. Defaults to 3D.')

    parser.add_argument('--pca_strength', type=float, default=1,
                        help='Strength of PCA augmentation')
    parser.add_argument('--augment', type=str, default='none',
                        help='augmentations')
    parser.add_argument('--M', type=int, default=10,
                        help='M')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='pretrain epochs')

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

    if args.dataset == 'Salinas':
        data_folder = 'Salinas'
    elif args.dataset == 'Pavia':
        data_folder = 'Pavia University'
    elif args.dataset == 'Indian':
        data_folder = 'Indian Pines'
    else:
        print('No dataset by right name')

    avg_acc = np.zeros(folds)
    writer = SummaryWriter('results/{}/overall/{}/'.format(args.run_name, args.augment))

    writer.add_text('Arguments', str(args))

    for f in range(0,folds):
        for r in range(args.runs):
            print('Running: ' + str(r) + 'time and: ' + str(f) + ' fold.')
            if args.method == 'supervised':
                if args.augment == 'none':
                    supervised_args = ['--model', args.model, '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                elif args.augment == 'spatial_combinations':
                    supervised_args = ['--augmentation_magnitude', str(args.M), '--spatial_combinations', '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                elif args.augment == 'spectral_mean':
                    supervised_args = ['--augmentation_magnitude', str(args.M), '--spectral_mean', '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                elif args.augment == 'moving_average':
                    supervised_args = ['--augmentation_magnitude', str(args.M), '--moving_average', '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                elif args.augment == 'pca':
                    supervised_args = ['--augmentation_magnitude', str(args.M), '--pca_augmentation', '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                result = supervised(supervised_args)
            elif args.method == 'mixup':
                result = mixup(['--class_balancing', '--dataset', args.dataset,
                                '--data_dir', data_path.format(data_folder), '--results', 'results/{}/'.format(args.run_name),
                                '--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr),
                                '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f),
                                '--cuda', '0', '--sampling_fixed', args.sampling_fixed])
            elif args.method == 'fixmatch':
                result = fixmatch(['--model', args.model, '--pretrain', str(args.pretrain),
                                   '--augmentation_magnitude', str(args.M), '--class_balancing',
                                   '--dataset', args.dataset, '--data_dir', data_path.format(data_folder),
                                   '--results', 'results/{}/'.format(args.run_name),'--epochs', '{}'.format(args.epochs),
                                   '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size),
                                   '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed,
                                   '--threshold', '{}'.format(args.threshold)])
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

    writer.add_text('Average accuracy per fold', str(avg_acc))
    writer.add_text('Average accuracy for all folds', str(np.sum(avg_acc)/len(avg_acc)))

    writer.close()

if __name__ == '__main__':
    """
    fixed_sampling = ['False', 'True']
    model = ['3D', '1D']
    for f in fixed_sampling:
        for m in model:
            main(['--server', '--sampling_fixed', f, '--method', 'fixmatch', '--runs', str(3), '--epochs', str(30), '--model', m, '--dataset', 'Pavia'])
    """
    """
    aug = ['spatial_combinations', 'moving_average', 'spectral_mean']
    M = [10, 12, 15, 20]
    for c in aug:
        for m in M:
            main(['--server', '--runs', str(3), '--augment', c, '--epochs', '10', '--M', str(m)])
    """

    """
    fixed_sampling = ['False', 'True']
    M = [2, 6, 10, 14]
    for f in fixed_sampling:
        for m in M:
            main(['--server', '--runs', str(3), '--epochs', str(50), '--method', 'fixmatch', '--sampling_fixed', f, '--M', str(m)])
    """

    """
    pretrain = ['5', '10', '15']
    for p in pretrain:
        main(['--server', '--runs', str(3), '--epochs', str(50), '--method', 'fixmatch', '--sampling_fixed', 'False', '--pretrain', p])
    pretrain = ['10', '20', '30']
    for p in pretrain:
        main(['--server', '--runs', str(3), '--epochs', str(100), '--method', 'fixmatch', '--sampling_fixed', 'True', '--pretrain', p])
    """
    main(['--server', '--runs', str(3), '--augment', 'pca', '--epochs', '10'])
