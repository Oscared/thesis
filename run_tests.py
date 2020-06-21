import os
import argparse
from fixmatch_hsi import main as fixmatch
from supervised_hsi import main as supervised
from mixup_hsi import main as mixup


def main():
    parser = argparse.ArgumentParser(description='Main testing file for running several tests over several datasets')
    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='Name of dataset to run. Salinas, PaviaU or Indian, defaults to Salinas')
    parser.add_argument('--fold', type=int, default=-1,
                        help='Specify one fold to run on patch datasets. Defaults to running on all.')
    parser.add_argument('--runs', type=int, default=1,
                        help='Amount of time to run on each dataset/fold. Defaults to 1.')
    parser.add_argument('--data_sampling', type=str, default='nalepa',
                        help='What kind of sampling of data to run. random, disjoint, region or patch based. Defaults to patch (nalepa).')
    parser.add_argument('--run_name', type=str, default='mixup',
                        help='Folder name to save all the results to. Defaults to results/fixmatch/')
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

    args = parser.parse_args()

    results = []

    if args.server:
        data_path = '/data/ieee_supplement/Hyperspectral_Grids/{}'
    else:
        data_path = '/home/oscar/Desktop/Exjobb/Data/ieee_supplement/Hyperspectral_Grids/{}'

    for f in range(0,5):
        for r in range(args.runs):
            print('Running: ' + str(r) + 'time and: ' + str(f) + ' fold.')
            if args.method == 'supervised':
                result = supervised(['--data_dir', data_path.format(args.dataset), '--results', 'results/{}/'.format(args.run_name),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--flip_augmentation'])
            elif args.method == 'mixup':
                result = mixup(['--data_dir', data_path.format(args.dataset), '--results', 'results/{}/'.format(args.run_name),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--flip_augmentation'])
            elif args.method == 'fixmatch':
                result = fixmatch(['--data_dir', data_path.format(args.dataset), '--results', 'results/{}/'.format(args.run_name),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--flip_augmentation'])
            else:
                print('No method with this name')
                results = None
            results.append(result)

    print('Ran all the folds.')


if __name__ == '__main__':
    main()
