import visdom
from datasets import get_dataset, HyperX
import utils
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from torch.nn import init
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
from torchsummary import summary
from torch.utils.tensorboard  import SummaryWriter

import math
import os
import datetime
from sklearn.externals import joblib
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Hyperspectral image classification with FixMatch")
    parser.add_argument('--patch_size', type=int, default=5,
                        help='Size of patch around each pixel taken for classification')
    parser.add_argument('--center_pixel', action='store_true', default=False,
                        help='use if you only want to consider the label of the center pixel of a patch')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of each batch for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of total epochs of training to run')
    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='Name of dataset to run, Salinas or PaviaU')
    parser.add_argument('--device', type=str, default='cpu',
                        help='what device to run on, cpu or gpu')
    parser.add_argument('--warmup', type=float, default=0,
                        help='warmup epochs')
    parser.add_argument('--save', action='store_true', default=False,
                        help='use to save model weights when running')
    parser.add_argument('--test_stride', type=int, default=3,
                        help='length of stride when sliding patch window over image for testing')
    parser.add_argument('--sampling_percentage', type=float, default=0.3,
                        help='percentage of dataset to sample for training (labeled and unlabeled included)')
    parser.add_argument('-sampling_mode', type=str, default='disjoint',
                        help='how to sample data, disjoint, random, or fixed')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='initial learning rate')
    parser.add_argument('--class_balancing', action='store_true', default=True,
                        help='use to balance weights according to ratio in dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='use to load model weights from a certain directory')
    parser.add_argument('--flip_augmentation', action='store_true', default=True,
                        help='use to flip augmentation data for use')
    parser.add_argument('--radiation_noise', action='store_true', default=True,
                        help='use to radiation noise data for use')
    parser.add_argument('--mixture_noise', action='store_true', default=True,
                        help='use to mixture noise data for use')
    parser.add_argument('--results', type=str, default='results',
                        help='where to save results to')
    parser.add_argument('--alpha', type=float, default=1,
                        help='mixup interpolation coefficient (default 1)')

    parser.add_argument('--supervision', type=str, default='full',
                        help='check this more, use to make us of all labeled or not, full or semi')

    args = parser.parse_arg()

    vis = visdom.Visdom()

    use_cude = torch.cuda.is_available()

    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(args.out)

    img, gt, label_values, ignored_labels, rgb_bands, palette = get_dataset(args.dataset)

    args.n_classes = len(label_values)
    args.n_bands = img.shape[-1]
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

    train_gt, test_gt = utils.sample_gt(gt, args.sampling_percentage,
                                        mode=args.sampling_mode)
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))

    utils.display_predictions(convert_to_color(train_gt), vis,
                              caption="Train ground truth")
    utils.display_predictions(convert_to_color(test_gt), vis,
                              caption="Test ground truth")

    device = torch.device(args.device)

    weights = torch.ones(args.n_classes)
    weights[torch.LongTensor(args.ignored_labels)] = 0
    weights = weights.to(device)

    model = HamidaEtAl(args.n_bands, args.n_classes,
                       patch_size=args.patch_size)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          nesterov=True)
    loss = nn.CrossEntropyLoss(weight=weights)

    train_gt, val_gt = utils.sample_gt(train_gt, 0.95, mode=args.sampling_mode)

    val_dataset = HyperX(img, val_gt, labeled=True, **args)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=args.batch_size)

    samples = np.count_nonzero(train_gt)

    train_labeled_gt, train_unlabeled_gt = utils.sample_gt(train_gt, 1/(args.unlabeled_ratio + 1),
                                                           mode=args.sampling_mode)

    train_labeled_dataset = HyperX(img, train_labeled_gt, labeled=True, **args)
    train_labeled_loader = data.DataLoader(train_labeled_dataset, batch_size=args.batch_size,
                                   shuffle=True, drop_last=True)

    train_unlabeled_dataset = HyperX(img, train_unlabeled_gt, labeled=False, **args)
    train_unlabeled_loader = data.DataLoader(train_unlabeled_dataset,
                                             batch_size=args.batch_size*unlabeled_portion,
                                             shuffle=True, drop_last=True)


    amount_labeled = samples // (args.unlabeled_ratio + 1)

    iterations = amount_labeled // args.batch_size
    total_steps = iterations * args.epochs
    args.scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                     args.warmup*iterations,
                                                     total_steps)

    utils.display_predictions(convert_to_color(train_labeled_gt), vis,
                              caption="Labeled train ground truth")
    utils.display_predictions(convert_to_color(train_unlabeled_gt), vis,
                              caption="Unlabeled train ground truth")
    utils.display_predictions(convert_to_color(val_gt), vis,
                              caption="Validation ground truth")

    if args.class_balancing:
        weights_balance = utils.compute_imf_weights(train_gt, args.n_classes,
                                                    args.ignored_labels)
        hyperparams['weights'] = torch.from_numpy(weights_balance)

    print(args)
    print("Network :")
    with torch.no_grad():
        for input, _ in train_labeled_loader:
            break
        summary(model.to(device), input.size()[1:])
        # We would like to use device=hyperparams['device'] altough we have
        # to wait for torchsummary to be fixed first.

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    try:
        train(model, optimizer, loss, train_labeled_loader,
              train_unlabeled_loader, args.epochs, writer=writer,
              scheduler=args.scheduler, device=args.device,
              threshold=args.threshold, val_loader=val_loader, display=vis,
              save=args.save)
    except KeyboardInterrupt:
        # Allow the user to stop the training
        pass

    probabilities = test(model, img, args)
    prediction = np.argmax(probabilities, axis=-1)

    run_results = utils.metrics(prediction, test_gt,
                                ignored_labels=args.ignored_labels,
                                n_classes=args.n_classes)

    mask = np.zeros(gt.shape, dtype='bool')
    for l in args.ignored_labels:
        mask[gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    utils.display_predictions(color_prediction, vis, gt=convert_to_color(test_gt),
                              caption="Prediction vs. test ground truth")

    utils.show_results(run_results, vis, label_values=label_values)

    writer.close()

def train(net, optimizer, criterion, train_loader, epoch,
          writer, scheduler=None, display_iter=100, device='cpu',
          display=None, val_loader=None, save=False):
    """
    Training loop to optimize a network for several epochs and a specified loss
    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        train_loader: a PyTorch dataset loader for the labeled dataset
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    device = torch.device(device)

    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1


    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.

        losses_meter = AverageMeter()

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

            batch_size = data.shape[0]
            # Load the data into the GPU if required
            data, targets = data.to(device), targets.to(device)

            data, target_a, target_b, lam = mixup_data(data, targets, args.alpha, use_cuda)
            data, target_a, target_b = map(Variable, (data, target_a, target_b))

            logits = net(data)

            loss = mixup_criterion(criterion, logits, target_a, target_b, lam)

            _, predicted = torch.max(logits.data, 1)
            mask += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_meter.update(loss.data[0])

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(e, epoch, batch_idx * len(data_x),
                                       len(data_x) * len(labeled_data_loader),
                                       100. * batch_idx / len(labeled_data_loader),
                                       mean_losses[iter_])
                update = None if loss_win is None else 'append'
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",
                          'ylabel': "Loss"
                         }
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                })
            iter_ += 1

        # Update the scheduler
        avg_loss /= len(labeled_data_loader)
        if val_loader is not None:
            val_acc, val_loss = val(net, val_loader, device=device, supervision='full')
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        writer.add_scalar('train/1.train_loss', losses_meter.avg, e)
        writer.add_scalar('test/1.test_acc', val_acc, e)
        writer.add_scalar('test/2.test_loss', val_loss.avg, e)

        # Save the weights
        if e % save_epoch == 0 && save == True:
            save_model(net, utils.camel_to_snake(str(net.__class__.__name__)),
                       labeled_data_loader.dataset.name, epoch=e, metric=abs(metric))


def val(net, data_loader, device='cpu', supervision='full'):
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
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            loss = F.cross_entropy(output, target)
            val_loss.update(loss.item())
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    val_acc.update(accuracy)
                    total += 1
                    """
                    Check this more to see if the loss is correct!
                    """
    return accuracy / total, val_loss


def save_model(model, model_name, dataset_name, **kwargs):
    """
    Save the models weights to a certain folder.
    """
     model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
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
            output = output.to(args.device)

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
        self.avg = self.sum / self.count


def mixup_data(x, y, alpha=1.0, use_cuda=False):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Returns the mixup loss based on a certain loss function (criterion)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
