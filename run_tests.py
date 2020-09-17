import os
import argparse
from fixmatch_hsi import main as fixmatch
from supervised_hsi import main as supervised
from mixup_hsi import main as mixup
from teacher_student_hsi import main as mean
import numpy as np
from tensorboardX import SummaryWriter
import datetime

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

    parser.add_argument('--n', type=int, default=0,
                        help='Amount of augmentations')
    parser.add_argument('--M', type=int, default=2,
                        help='Strength of augmentations')

    parser.add_argument('--pretrain', type=int, default=0,
                        help='pretrain epochs')
    parser.add_argument('--extra_data', type=str, default='True',
                        help='extra data for pavia. Defaults to true.')
    parser.add_argument('--samples', type=int, default=10,
                        help='samples per class when fixed sampling. Defaults to 10.')
    parser.add_argument('--unlabeled_ratio', type=int, default=7,
                        help='Ratio of unlabeled samples per batch. Defaults to 7.')

    parser.add_argument('--warmup', type=float, default=0,
                        help='warmup epochs')
    parser.add_argument('--consistency', type=float, default=100.0,
                        help='Consistency weight maximum value. Defaults to 100.')
    parser.add_argument('--consistency_rampup', type=int, default=5,
                        help='epochs of rampup for consistency? Defaults to 5.')
    parser.add_argument('--ema_decay', type=float, default=0.95,
                        help='EMA decay of weights. Defaults to 0.95.')


    args = parser.parse_args(raw_args)

    results = []

    if args.server:
        data_path = '/data/ieee_supplement/Hyperspectral_Grids/{}'
    else:
        data_path = '/home/oscar/Desktop/Exjobb/Data/ieee_supplement/Hyperspectral_Grids/{}'

    if args.dataset == 'Indiana':
        folds = 4
    else:
        folds = 5

    if args.dataset == 'Salinas':
        data_folder = 'Salinas'
    elif args.dataset == 'Pavia':
        data_folder = 'Pavia University'
    elif args.dataset == 'Indiana':
        data_folder = 'Indian Pines'
    else:
        print('No dataset by right name')

    avg_acc = np.zeros(folds)

    tensorboard_dir = 'results/{}/overall/{}/'.format(args.run_name, datetime.datetime.now().strftime("%m-%d-%X"))

    os.makedirs(tensorboard_dir, exist_ok=True)

    writer = SummaryWriter(tensorboard_dir)

    writer.add_text('Arguments', str(args))

    for f in range(0,folds):
        for r in range(args.runs):
            print('Running: ' + str(r) + 'time and: ' + str(f) + ' fold.')
            if args.method == 'supervised':
                if args.augment == 'none':
                    supervised_args = ['--model', args.model, '--class_balancing',
                                       '--dataset', args.dataset, '--data_dir', data_path.format(data_folder),
                                       '--results', 'results/{}/{}/'.format(args.run_name, args.augment), '--epochs', '{}'.format(args.epochs),
                                       '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size),
                                       '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed,
                                       '--samples_per_class', str(args.samples)]

                elif args.augment == 'rand':
                    supervised_args = ['--model', args.model, '--class_balancing',
                                       '--dataset', args.dataset, '--data_dir', data_path.format(data_folder),
                                       '--results', 'results/{}/{}/{}/{}'.format(args.run_name, args.augment, args.n, args.M),
                                       '--epochs', '{}'.format(args.epochs), '--augmentation_amount', str(args.n),
                                       '--augmentation_magnitude', str(args.M),
                                       '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size),
                                       '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed,
                                       '--samples_per_class', str(args.samples)]
                elif args.augment == 'spatial_combinations':
                    supervised_args = ['--augmentation_magnitude', str(args.M), '--spatial_combinations', '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                elif args.augment == 'spectral_mean':
                    supervised_args = ['--augmentation_magnitude', str(args.M), '--spectral_mean', '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                elif args.augment == 'moving_average':
                    supervised_args = ['--augmentation_magnitude', str(args.M), '--moving_average', '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                elif args.augment == 'pca':
                    supervised_args = ['--pca_strength', str(args.M), '--pca_augmentation', '--class_balancing', '--dataset', args.dataset, '--data_dir', data_path.format(data_folder), '--results', 'results/{}/{}/'.format(args.run_name, args.augment),'--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed]
                result = supervised(supervised_args)
            elif args.method == 'mixup':
                result = mixup(['--class_balancing', '--dataset', args.dataset,
                                '--data_dir', data_path.format(data_folder), '--results', 'results/{}/'.format(args.run_name),
                                '--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr),
                                '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f),
                                '--cuda', '0', '--sampling_fixed', args.sampling_fixed])
            elif args.method == 'mean':
                result = mean(['--class_balancing', '--dataset', args.dataset,
                                '--data_dir', data_path.format(data_folder), '--results', 'results/{}/'.format(args.run_name),
                                '--epochs', '{}'.format(args.epochs), '--lr', '{}'.format(args.lr),
                                '--batch_size', '{}'.format(args.batch_size), '--fold', '{}'.format(f),
                                '--cuda', '0', '--sampling_fixed', args.sampling_fixed,
                                '--extra_data', args.extra_data, '--unlabeled_ratio', str(args.unlabeled_ratio),
                                '--samples_per_class', str(args.samples), '--model', args.model,
                                '--warmup', str(args.warmup), '--consistency', str(args.consistency),
                                '--consistency_rampup', str(args.consistency_rampup), '--ema_decay', str(args.ema_decay),
                                '--augmentation_magnitude', str(args.M), '--augmentation_amount', str(args.n)])
            elif args.method == 'fixmatch':
                result = fixmatch(['--model', args.model, '--pretrain', str(args.pretrain),
                                   '--augmentation_magnitude', str(args.M), '--augmentation_amount', str(args.n),
                                   '--class_balancing', '--flip_augmentation',
                                   '--dataset', args.dataset, '--data_dir', data_path.format(data_folder),
                                   '--results', 'results/{}/{}'.format(args.run_name, str(args.samples)),'--epochs', '{}'.format(args.epochs),
                                   '--lr', '{}'.format(args.lr), '--batch_size', '{}'.format(args.batch_size),
                                   '--fold', '{}'.format(f), '--cuda', '0', '--sampling_fixed', args.sampling_fixed,
                                   '--threshold', '{}'.format(args.threshold), '--extra_data', args.extra_data,
                                   '--samples_per_class', str(args.samples), '--unlabeled_ratio', str(args.unlabeled_ratio)])
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
    method = ['mean', 'fixmatch', 'supervised']
    extra_data = ['False', 'True']
    dataset = ['Pavia', 'Salinas']
    sampling = ['True', 'False']
    for m in method:
        for s in sampling:
            for d in dataset:
                if d == 'Pavia':
                    for e in extra_data:
                        main(['--server', '--sampling_fixed', s, '--method', m, '--runs', str(2),
                              '--epochs', str(60), '--dataset', d, '--extra_data', e, '--samples', str(40),
                              '--run_name', 'method_comparision_1D', '--model', '1D'])
                else:
                    main(['--server', '--sampling_fixed', s, '--method', m, '--runs', str(2),
                          '--epochs', str(60), '--dataset', d, '--samples', str(40),
                          '--run_name', 'method_comparision_1D', '--model', '1D'])
    """
    """ KÖR SEN
    sampling = ['True', 'False']
    extra_data = ['True', 'False']
    dataset = ['Pavia', 'Salinas']
    for s in sampling:
        for d in dataset:
            if d == 'Pavia':
                for e in extra_data:
                    main(['--server', '--sampling_fixed', s, '--method', 'mean', '--runs', str(2),
                          '--epochs', str(60), '--dataset', d, '--extra_data', e, '--samples', str(40)])
            else:
                main(['--server', '--sampling_fixed', s, '--method', 'mean', '--runs', str(2),
                      '--epochs', str(60), '--dataset', d, '--samples', str(40)])
    """

    #aug = ['none', 'spatial_combinations', 'moving_average', 'spectral_mean', 'pca']
    N = [1,2,4]
    M = [6, 8, 10]
    for n in N:
        for m in M:
            main(['--server', '--method', 'fixmatch', '--runs', str(2), '--epochs',
                  '60', '--n', str(n), '--M', str(m), '--run_name', 'fixmatch_rand_aug/{}/{}'.format(n,m),
                  '--sampling_fixed', 'True', '--samples', str(40)])
    N = [1]
    M = [6, 10]
    for n in N:
        for m in M:
            main(['--server', '--method', 'mean', '--runs', str(2), '--epochs',
                  '60', '--n', str(n), '--M', str(m), '--run_name', 'mean_rand_aug/{}/{}'.format(n,m),
                  '--sampling_fixed', 'True', '--samples', str(40)])

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

    """ Good param is warmup 0, consistency 100, ramp_up 5, decay 0.9 for example
    warmup = ['5', '10']
    consistency = ['80', '100', '120']
    ramp_up = ['2', '5', '7']
    decay = ['0.90', '0.95', '0.99']
    for w in warmup:
        for c in consistency:
            for r in ramp_up:
                for d in decay:
                    main(['--server', '--sampling_fixed', 'True', '--method', 'mean', '--runs', str(2),
                          '--epochs', str(60), '--dataset', 'Salinas', '--samples', str(40),
                          '--run_name', 'mean_param_test', '--warmup', w,
                          '--consistency', c, '--consistency_rampup', r,
                          '--ema_decay', d])
    """
