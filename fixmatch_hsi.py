import visdom
from datasets import get_dataset, get_patch_data, get_pixel_idx, HyperX, HyperX_patches
import utils
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from torch.nn import init
import torch.utils.data as data
#from torchsummary import summary
#from torch.utils.tensorboard  import SummaryWriter
from tensorboardX import SummaryWriter
from models import NalepaEtAl

import math
import os
import datetime
#import joblib
from tqdm import tqdm
import argparse

def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Hyperspectral image classification with FixMatch")
    parser.add_argument('--patch_size', type=int, default=5,
                        help='Size of patch around each pixel taken for classification')
    parser.add_argument('--center_pixel', action='store_false',
                        help='use if you only want to consider the label of the center pixel of a patch')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Size of each batch for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of total epochs of training to run')
    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='Name of dataset to run, Salinas or PaviaU')
    parser.add_argument('--cuda', type=int, default=-1,
                        help='what CUDA device to run on, -1 defaults to cpu')
    parser.add_argument('--warmup', type=float, default=0,
                        help='warmup epochs')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='confidence threshold for pseudo labels')
    parser.add_argument('--save', action='store_true',
                        help='use to save model weights when running')
    parser.add_argument('--test_stride', type=int, default=1,
                        help='length of stride when sliding patch window over image for testing')
    parser.add_argument('--sampling_percentage', type=float, default=0.3,
                        help='percentage of dataset to sample for training (labeled and unlabeled included)')
    parser.add_argument('--sampling_mode', type=str, default='nalepa',
                        help='how to sample data, disjoint, random, or fixed')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--unlabeled_ratio', type=int, default=7,
                        help='ratio of unlabeled data to labeled (spliting the training data into these ratios)')
    parser.add_argument('--class_balancing', action='store_false',
                        help='use to balance weights according to ratio in dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='use to load model weights from a certain directory')
    parser.add_argument('--model', type=str, default='3D',
                        help='Choose model. Possible is 3D or 1D. Defaults to 3D.')
    #Augmentation arguments
    parser.add_argument('--flip_augmentation', action='store_true',
                        help='use to flip augmentation data for use')
    parser.add_argument('--radiation_augmentation', action='store_true',
                        help='use to radiation noise data for use')
    parser.add_argument('--mixture_augmentation', action='store_true',
                        help='use to mixture noise data for use')
    parser.add_argument('--pca_augmentation', action='store_true',
                        help='use to pca augment data for use')
    parser.add_argument('--pca_strength', type=float, default=1.0,
                        help='Strength of the PCA augmentation, defaults to 1.')
    parser.add_argument('--cutout_spatial', action='store_true',
                        help='use to cutout spatial for data augmentation')
    parser.add_argument('--cutout_spectral', action='store_true',
                        help='use to cutout spectral for data augmentation')
    parser.add_argument('--spatial_combinations', action='store_true',
                        help='use to spatial combine for data augmentation')
    parser.add_argument('--spectral_mean', action='store_true',
                        help='use to spectal mean for data augmentation')
    parser.add_argument('--moving_average', action='store_true',
                        help='use to sprectral moving average for data augmentation')

    parser.add_argument('--augmentation_magnitude', type=int, default=3,
                        help='Magnitude of augmentation (so far only for cutout). Defualts to 1, min 1 and max 10.')
    parser.add_argument('--augmentation_amount', type=int, default=2,
                        help='amount of augmentation (so far only for cutout). Defualts to 1, min 1 and max 10.')


    parser.add_argument('--results', type=str, default='results',
                        help='where to save results to (default results)')
    parser.add_argument('--save_dir', type=str, default='/saves/',
                        help='where to save models to (default /saves/)')
    parser.add_argument('--data_dir', type=str, default='/data/',
                        help='where to fetch data from (default /data/)')
    parser.add_argument('--load_file', type=str, default=None,
                        help='wihch file to load weights from (default None)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold to sample from if using Nalepas validation scheme')
    parser.add_argument('--sampling_fixed', type=str, default='True',
                        help='Use to sample a fixed amount of samples for each class from Nalepa sampling')
    parser.add_argument('--samples_per_class', type=int, default=10,
                        help='Amount of samples to sample for each class when sampling a fixed amount. Defaults to 10.')


    parser.add_argument('--pretrain', type=int, default=0,
                        help='amount of epochs to train on only labeled samples. Defaults to 0.')
    parser.add_argument('--supervision', type=str, default='full',
                        help='check this more, use to make us of all labeled or not, full or semi')
    parser.add_argument('--extra_data', type=str, default='True',
                        help='add extra data for pavia dataset. Defaults to true.')

    args = parser.parse_args(raw_args)

    device = utils.get_device(args.cuda)
    args.device = device

    #vis = visdom.Visdom()
    vis = None

    tensorboard_dir = str(args.results + '/' + datetime.datetime.now().strftime("%m-%d-%X"))

    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    if args.model == '1D':
        args.patch_size = 1

    if args.sampling_mode == 'nalepa':
        train_img, train_gt, test_img, test_gt, label_values, ignored_labels, rgb_bands, palette = get_patch_data(args.dataset, args.patch_size, target_folder=args.data_dir, fold=args.fold)
        args.n_bands = train_img.shape[-1]

        if args.dataset == 'Pavia' and args.extra_data == 'True':
            print('Extra data Pavia')
            train_img, train_gt, test_img, test_gt, label_values, ignored_labels, rgb_bands, palette = get_patch_data('pavia', args.patch_size, target_folder=args.data_dir, fold=args.fold)
            args.n_bands = train_img.shape[-1]
            img_unlabeled, _, _, _, _, _ = get_dataset('PaviaC', target_folder='/data/')
            gt_unlabeled = np.zeros((img_unlabeled.shape[0], img_unlabeled.shape[1]))
            gt_1 = gt_unlabeled[:,:223]
            gt_2 = gt_unlabeled[:,223:]
            img_1 = img_unlabeled[:,:223,:]
            img_2 = img_unlabeled[:,223:,:]

            pad_width = args.patch_size // 2

            img_1 = np.pad(img_1, ((pad_width, pad_width), (pad_width, pad_width), (0,0)))
            gt_1 = np.pad(gt_1, ((pad_width, pad_width), (pad_width, pad_width)))

            img_2 = np.pad(img_2, ((pad_width, pad_width), (pad_width, pad_width), (0,0)))
            gt_2 = np.pad(gt_2, ((pad_width, pad_width), (pad_width, pad_width)))

            img_1 = img_1.reshape(1, img_1.shape[0], img_1.shape[1], img_1.shape[2])
            idx_1 = np.array([(0,x,y) for x in range(pad_width,gt_1.shape[0]-pad_width) for y in range(pad_width,gt_1.shape[1]-pad_width)])

            img_2 = img_2.reshape(1, img_2.shape[0], img_2.shape[1], img_2.shape[2])
            idx_2 = np.array([(0,x,y) for x in range(pad_width,gt_2.shape[0]-pad_width) for y in range(pad_width,gt_2.shape[1]-pad_width)])

            img_1 = np.concatenate((img_1, img_1[:,:,:,-1, np.newaxis]), axis=-1)
            img_2 = np.concatenate((img_2, img_2[:,:,:,-1, np.newaxis]), axis=-1)
    else:
        img, gt, label_values, ignored_labels, rgb_bands, palette = get_dataset(args.dataset, target_folder=args.data_dir)
        args.n_bands = img.shape[-1]

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

    if args.sampling_mode == 'nalepa':
        print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(train_gt) + np.count_nonzero(test_gt)))
        writer.add_text('Amount of training samples', "{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(test_gt)))

        utils.display_predictions(convert_to_color(test_gt), vis, writer=writer,
                                  caption="Test ground truth")
    else:
        train_gt, test_gt = utils.sample_gt(gt, args.sampling_percentage,
                                            mode=args.sampling_mode)
        print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
        writer.add_text('Amount of training samples', "{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))

        utils.display_predictions(convert_to_color(train_gt), vis, writer=writer,
                                    caption="Train ground truth")
        utils.display_predictions(convert_to_color(test_gt), vis, writer=writer,
                                    caption="Test ground truth")

    if args.model == '3D':
        model = HamidaEtAl(args.n_bands, args.n_classes,
                           patch_size=args.patch_size)
    if args.model == '1D':
        model = NalepaEtAl(args.n_bands, args.n_classes)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          nesterov=True, weight_decay=0.0005)
    #loss_labeled = nn.CrossEntropyLoss(weight=weights)
    #loss_unlabeled = nn.CrossEntropyLoss(weight=weights, reduction='none')

    if args.sampling_mode == 'nalepa':
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

        val_dataset = HyperX_patches(train_img, train_gt, idx_val, labeled='Val', **vars(args))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)

        train_labeled_dataset = HyperX_patches(train_img, train_gt, idx_sup, labeled=True, **vars(args))
        train_labeled_loader = data.DataLoader(train_labeled_dataset, batch_size=args.batch_size,
                                       pin_memory=True, num_workers=10,
                                       shuffle=True, drop_last=True)

        #unlabeled_ratio = math.ceil(len(idx_unsup)/len(idx_sup))
        unlabeled_ratio = args.unlabeled_ratio

        train_unlabeled_dataset = HyperX_patches(train_img, train_gt, idx_unsup, labeled=False, **vars(args))
        if args.dataset == 'Pavia' and args.extra_data == 'True':
            train_unlabeled_dataset = data.ConcatDataset([#train_unlabeled_dataset,
                                                          HyperX_patches(img_1, gt_1, idx_1, labeled=False, **vars(args)),
                                                          HyperX_patches(img_2, gt_2, idx_2, labeled=False, **vars(args))])
        train_unlabeled_loader = data.DataLoader(train_unlabeled_dataset, batch_size=args.batch_size*unlabeled_ratio,
                                       pin_memory=True, num_workers=10,
                                       shuffle=True, drop_last=True)

        amount_labeled = idx_sup.shape[0]
    else:
        train_gt, val_gt = utils.sample_gt(train_gt, 0.95, mode=args.sampling_mode)

        val_dataset = HyperX(img, val_gt, labeled='Val', **vars(args))
        val_loader = data.DataLoader(val_dataset,
                                    batch_size=args.batch_size)

        train_labeled_gt, train_unlabeled_gt = utils.sample_gt(train_gt, 1/(args.unlabeled_ratio + 1),
                                                                mode=args.sampling_mode)

        writer.add_text('Amount of labeled training samples', "{} samples selected (over {})".format(np.count_nonzero(train_labeled_gt), np.count_nonzero(train_gt)))
        writer.add_text('Amount of unlabeled training samples', "{} samples selected (over {})".format(np.count_nonzero(train_unlabeled_gt), np.count_nonzero(train_gt)))
        samples_class = np.zeros(args.n_classes)
        for c in np.unique(train_labeled_gt):
            samples_class[c-1] = np.count_nonzero(train_labeled_gt == c)
        writer.add_text('Labeled samples per class', str(samples_class))

        train_labeled_dataset = HyperX(img, train_labeled_gt, labeled=True, **vars(args))
        train_labeled_loader = data.DataLoader(train_labeled_dataset, batch_size=args.batch_size,
                                                pin_memory=True, num_workers=5,
                                                shuffle=True, drop_last=True)

        train_unlabeled_dataset = HyperX(img, train_unlabeled_gt, labeled=False, **vars(args))
        train_unlabeled_loader = data.DataLoader(train_unlabeled_dataset,
                                                    batch_size=args.batch_size*args.unlabeled_ratio,
                                                    pin_memory=True, num_workers=5,
                                                    shuffle=True, drop_last=True)


        amount_labeled = np.count_nonzero(train_labeled_gt)

        utils.display_predictions(convert_to_color(train_labeled_gt), vis, writer=writer,
                                  caption="Labeled train ground truth")
        utils.display_predictions(convert_to_color(train_unlabeled_gt), vis, writer=writer,
                                  caption="Unlabeled train ground truth")
        utils.display_predictions(convert_to_color(val_gt), vis, writer=writer,
                                  caption="Validation ground truth")

    args.iterations = amount_labeled // args.batch_size
    args.total_steps = args.iterations * args.epochs
    args.scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                     args.warmup*args.iterations,
                                                     args.total_steps)


    if args.class_balancing:
        weights_balance = utils.compute_imf_weights(train_gt, len(label_values),
                                                    args.ignored_labels)
        args.weights = torch.from_numpy(weights_balance[1:])
        args.weights = args.weights.to(torch.float32)
    else:
        weights = torch.ones(args.n_classes)
        #weights[torch.LongTensor(args.ignored_labels)] = 0
        args.weights = weights

    args.weights = args.weights.to(args.device)
    loss_labeled = nn.CrossEntropyLoss(weight=args.weights)
    loss_unlabeled = nn.CrossEntropyLoss(weight=args.weights, reduction='none')
    loss_val = nn.CrossEntropyLoss(weight=args.weights)

    print(args)
    print("Network :")
    writer.add_text('Arguments', str(args))
    with torch.no_grad():
        for input, _ in train_labeled_loader:
            break
        #summary(model.to(device), input.size()[1:])
        #writer.add_graph(model.to(device), input)
        # We would like to use device=hyperparams['device'] altough we have
        # to wait for torchsummary to be fixed first.

    if args.load_file is not None:
        model.load_state_dict(torch.load(args.load_file))
    model.zero_grad()

    try:
        train(model, optimizer, loss_labeled, loss_unlabeled, loss_val, train_labeled_loader,
              train_unlabeled_loader, writer, args, val_loader=val_loader, display=vis)
    except KeyboardInterrupt:
        # Allow the user to stop the training
        pass

    if args.sampling_mode=='nalepa':
        probabilities = test(model, test_img, args)
    else:
        probabilities = test(model, img, args)

    prediction = np.argmax(probabilities, axis=-1)

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

def train(net, optimizer, criterion_labeled, criterion_unlabeled, criterion_val, labeled_data_loader,
          unlabeled_data_loader, writer, args,
          display_iter=10, display=None, val_loader=None):
    """
    Training loop to optimize a network for several epochs and a specified loss
    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        labeled_data_loader: a PyTorch dataset loader for the labeled dataset
        unlabeled_data_loader: a PyTorch dataset loader for the weakly and
                               strongly augmented, unlabeled dataset
        epoch: int specifying the number of training epochs
        threshold: probability threshold for pseudo labels acceptance
        criterion_labeled: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
                            for the labeled Training
        criterion_unlabeled: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
                            for the unlabeled training
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    save_dir = args.save_dir


    if criterion_labeled is None or criterion_unlabeled is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(args.device)

    save_epoch = args.epochs // 20 if args.epochs > 20 else 1


    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, args.epochs + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.

        losses_meter = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        train_loader = zip(labeled_data_loader, unlabeled_data_loader)

        # Run the training loop for one epoch
        for batch_idx, (data_x, data_u) in tqdm(enumerate(train_loader), total=len(unlabeled_data_loader)):
            inputs_x, targets_x = data_x
            #Try to remove prediction for ignored class
            targets_x = targets_x - 1

            inputs_u_w, inputs_u_s = data_u

            batch_size = inputs_x.shape[0]
            # Load the data into the GPU if required
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
            targets_x = targets_x.to(args.device)
            logits = net(inputs)
            logits_x = logits[:args.batch_size]
            logits_u_w, logits_u_s = logits[args.batch_size:].chunk(2)
            del logits

            Lx = criterion_labeled(logits_x, targets_x)

            pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            #To see that we don't predict any ignored labels
            #ignored_index = np.nonzero(targets_u==args.ignored_labels)
            mask = max_probs.ge(args.threshold).float()
            #mask[ignored_index] = 0

            Lu = (criterion_unlabeled(logits_u_s, targets_u) * mask).mean()

            if e < args.pretrain:
                loss = Lx
            else:
                loss = Lx + 1 * Lu


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            args.scheduler.step()

            losses_meter.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            mask_prob = mask.mean().item()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [Iter: {:4}/{:4}]\tLr: {:.6f}\tLoss: {:.4f}\tLoss_labeled: {:.4f}\tLoss_unlabeled: {:.4f}\tMask: {:.4f}'
                string = string.format(e, args.epochs, batch_idx + 1,
                                       args.iterations, args.scheduler.get_last_lr()[0],
                                       losses_meter.avg, losses_x.avg, losses_u.avg, mask_prob)
                update = None if loss_win is None else 'append'
                if display is not None:
                    loss_win = display.line(
                        X=np.arange(iter_ - display_iter, iter_),
                        Y=mean_losses[iter_ - display_iter:iter_],
                        win=loss_win,
                        update=update,
                        opts={'title': "Training loss",
                            'xlabel': "Iterations",
                            'ylabel': "Loss"
                             })

                tqdm.write(string)
                '''
                for name, param in net.named_parameters():
                    if param.requires_grad:
                        print(name, torch.min(param.data), torch.max(param.data))
                '''
                if len(val_accuracies) > 0 and display is not None:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                })
            iter_ += 1
            del(data_x, data_u, loss, inputs_u_s, inputs_u_w, inputs_x, targets_x,
                logits_x, logits_u_s, logits_u_w, Lx, Lu)

        # Update the scheduler
        avg_loss /= len(labeled_data_loader)
        if val_loader is not None:
            val_acc, val_loss = val(net, val_loader, criterion_val, device=args.device, supervision='full')
            #val_accuracies.append(val_acc)
            #metric = -val_acc
        else:
            metric = avg_loss

        writer.add_scalar('train/1.train_loss', losses_meter.avg, e)
        writer.add_scalar('train/2.train_loss_x', losses_x.avg, e)
        writer.add_scalar('train/3.train_loss_u', losses_u.avg, e)
        writer.add_scalar('train/4.mask', mask_prob, e)
        writer.add_scalar('test/1.test_acc', val_acc.avg, e)
        writer.add_scalar('test/2.test_loss', val_loss.avg, e)

        # Save the weights
        if e % save_epoch == 0 and args.save == True:
            save_model(net, utils.camel_to_snake(str(net.__class__.__name__)),
                       labeled_data_loader.dataset.name, args.save_dir, epoch=args.epochs, metric=abs(-val_acc.avg))


def val(net, data_loader, loss_val, device=torch.device('cpu'), supervision='full'):
    """
    Validate the model on the validation dataset
    """
    # TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            # Try to remove prediction for ignored class
            target = target - 1
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            loss = loss_val(output, target)
            val_loss.update(loss.item())
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    val_acc.update(out.item() == pred.item())
                    total += 1
                    """
                    Check this more to see if the loss is correct!
                    """
    return val_acc, val_loss


def save_model(model, model_name, dataset_name, save_dir, **kwargs):
    """
    Save the models weights to a certain folder.
    """
    model_dir = save_dir + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = str(datetime.datetime.now()) + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + '.pth')
    else:
        filename = str(datetime.datetime.now())
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, args):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = args.patch_size
    center_pixel = args.center_pixel
    batch_size, device = args.batch_size, torch.device(args.device)
    n_classes = args.n_classes

    kwargs = {'step': args.test_stride, 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = utils.count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(utils.grouper(batch_size, utils.sliding_window(img, **kwargs)),
                      total=(iterations), desc="Inference on the image"):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0))

        self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=7./16., last_epoch=-1):
    """
    Training scheme for the learning rate with a cosine schedule with warmup
    """
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
