import os
import argparse
from supervised_hsi import main as supervised


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
    parser.add_argument('--run_name', type=str, default='supervised',
                        help='Folder name to save all the results to. Defaults to results/supervised/')

    args = parser.parse_args()

    for f in range(0,5):
        for r in range(args.runs):
            print('Running: ' + str(r) + 'time and: ' + str(f) + ' fold.')
            supervised(['--data_dir', '/home/oscar/Desktop/Exjobb/Data/ieee_supplement/Hyperspectral_Grids/{}'.format(args.dataset), '--results', 'results/{}/'.format(args.run_name),'--epochs', '2', '--lr', '0.001', '--batch_size', '10', '--fold={}'.format(f)])

    print('Ran all the folds.')


if __name__ == '__main__':
    main()
