import visdom
from datasets import get_dataset, get_patch_data, HyperX, HyperX_patches
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
from torch.autograd import Variable
import torch.utils.data as data
#from torchsummary import summary
#from torch.utils.tensorboard  import SummaryWriter
from tensorboardX import SummaryWriter

import math
import os
import datetime
#import joblib
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Hyperspectral image classification with FixMatch")
    parser.add_argument('--patch_size', type=int, default=5,
                        help='Size of patch around each pixel taken for classification')
    parser.add_argument('--center_pixel', action='store_false',
                        help='use if you only want to consider the label of the center pixel of a patch')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of each batch for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of total epochs of training to run')
    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='Name of dataset to run, Salinas or PaviaU')
    parser.add_argument('--cuda', type=int, default=-1,
                        help='what CUDA device to run on, -1 defaults to cpu')
    parser.add_argument('--warmup', type=float, default=0,
                        help='warmup epochs')
    parser.add_argument('--save', action='store_true',
                        help='use to save model weights when running')
    parser.add_argument('--test_stride', type=int, default=1,
                        help='length of stride when sliding patch window over image for testing')
    parser.add_argument('--sampling_percentage', type=float, default=0.3,
                        help='percentage of dataset to sample for training (labeled and unlabeled included)')
    parser.add_argument('--sampling_mode', type=str, default='nalepa',
                        help='how to sample data, disjoint, random, or fixed')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='initial learning rate')
    parser.add_argument('--class_balancing', action='store_false',
                        help='use to balance weights according to ratio in dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='use to load model weights from a certain directory')
    parser.add_argument('--flip_augmentation', action='store_true',
                        help='use to flip augmentation data for use')
    parser.add_argument('--radiation_augmentation', action='store_true',
                        help='use to radiation noise data for use')
    parser.add_argument('--mixture_augmentation', action='store_true',
                        help='use to mixture noise data for use')
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


    parser.add_argument('--supervision', type=str, default='full',
                        help='check this more, use to make us of all labeled or not, full or semi')

    args = parser.parse_args()

    device = utils.get_device(args.cuda)
    args.device = device

    if args.use_vis == True:
        vis = visdom.Visdom()
    else:
        vis = None

    tensorboard_dir = str(args.results + '/' + datetime.datetime.now().strftime("%m-%d-%X"))

    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    if args.sampling_mode == 'nalepa':
        train_img, train_gt, test_img, test_gt, label_values, ignored_labels, rgb_bands, palette = get_patch_data(args.dataset, args.patch_size, target_folder=args.data_dir)
        args.n_bands = train_img.shape[-1]
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

    model = HamidaEtAl(args.n_bands, args.n_classes,
                       patch_size=args.patch_size)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          nesterov=True)
    #loss_labeled = nn.CrossEntropyLoss(weight=weights)
    #loss_unlabeled = nn.CrossEntropyLoss(weight=weights, reduction='none')

    if args.sampling_mode == 'nalepa':
        #Get fixed amount of random samples for validation
        idx = set()
        while len(idx) < 0.05*np.prod(train_gt.shape):
            p = np.random.randint(train_gt.shape[0])
            x = np.random.randint(train_gt.shape[1])
            y = np.random.randint(train_gt.shape[2])
            if (p,x,y) not in idx:
                idx.add((p,x,y))

        #Training data
        train_labeled = train_img[idx]
        train_labeled_gt = train_gt[idx]
        #Validation data. Random subset of 95% of training data
        mask = np.zeros_like(train_img)
        mask[idx] = False
        val_img = train_img[mask]
        val_gt = train_gt[mask[0:3]]

        writer.add_text('Amount of labeled training samples', "{} samples selected (over {})".format(np.count_nonzero(train_labeled_gt), np.count_nonzero(train_gt)))

        samples_class = np.zeros(args.n_classes)
        for c in np.unique(train_labeled_gt):
            samples_class[c-1] = np.count_nonzero(train_labeled_gt == c)
        writer.add_text('Labeled samples per class', str(samples_class))

        val_dataset = HyperX_patches(val_img, val_gt, labeled=True, **vars(args))
        val_loader = data.Dataloader(val_dataset, batch_size=args.batch_size)

        train_dataset = HyperX_patches(train_labeled, train_labeled_gt, labeled=True, **vars(args))
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, num_workers=5,
                                       shuffle=True, drop_last=True)

    else:
        train_labeled_gt, val_gt = utils.sample_gt(train_gt, 0.95, mode=args.sampling_mode)

        val_dataset = HyperX(img, val_gt, labeled=True, **vars(args))
        val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)

        writer.add_text('Amount of labeled training samples', "{} samples selected (over {})".format(np.count_nonzero(train_labeled_gt), np.count_nonzero(train_gt)))
        samples_class = np.zeros(args.n_classes)
        for c in np.unique(train_labeled_gt):
            samples_class[c-1] = np.count_nonzero(train_labeled_gt == c)
        writer.add_text('Labeled samples per class', str(samples_class))

        train_dataset = HyperX(img, train_labeled_gt, labeled=True, **vars(args))
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, num_workers=5,
                                       shuffle=True, drop_last=True)

        utils.display_predictions(convert_to_color(train_labeled_gt), vis, writer=writer,
                                  caption="Labeled train ground truth")
        utils.display_predictions(convert_to_color(val_gt), vis, writer=writer,
                                  caption="Validation ground truth")

    amount_labeled = np.count_nonzero(train_labeled_gt)

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
    criterion = nn.CrossEntropyLoss(weight=args.weights)
    loss_val = nn.CrossEntropyLoss(weight=args.weights)

    print(args)
    print("Network :")
    writer.add_text('Arguments', str(args))
    with torch.no_grad():
        for input, _ in train_loader:
            break
        #summary(model.to(device), input.size()[1:])
        #writer.add_graph(model.to(device), input)
        # We would like to use device=hyperparams['device'] altough we have
        # to wait for torchsummary to be fixed first.

    if args.load_file is not None:
        model.load_state_dict(torch.load(args.load_file))
    model.zero_grad()

    try:
        train(model, optimizer, criterion, loss_val, train_loader,
              writer, args, val_loader=val_loader, display=vis)
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
    prediction += 1
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    utils.display_predictions(color_prediction, vis, gt=convert_to_color(test_gt), writer=writer,
                              caption="Prediction vs. test ground truth")

    utils.show_results(run_results, vis, writer=writer, label_values=label_values)

    writer.close()

def train(net, optimizer, criterion_labeled, criterion_val, train_loader,
          writer, args, display_iter=10, display=None, val_loader=None):
    """
    Training loop to optimize a network for several epochs and a specified loss
    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        train_loader: a PyTorch dataset loader for the labeled dataset
        epoch: int specifying the number of training epochs
        threshold: probability thresold for pseudo labels acceptance
        criterion_labeled: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    save_dir = args.save_dir


    if criterion_labeled is None:
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
        train_loss = 0
        correct = 0
        total = 0

        # Run the training loop for one epoch
        for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            targets = targets - 1

            batch_size = data.shape[0]
            # Load the data into the GPU if required
            data = data.to(args.device)
            targets = targets.to(args.device)

            inputs, targets_a, targets_b, lam = mixup_data(data, targets,
                                                           args.alpha, args.device)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                               targets_a, targets_b))

            outputs = net(inputs)
            loss = mixup_criterion(criterion_labeled, outputs, targets_a, targets_b, lam)

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).to('cpu').sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).to('cpu').sum().float())


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            args.scheduler.step()

            losses_meter.update(loss.item())

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [Iter: {:4}/{:4}]\tLr: {:.6f}\tLoss: {:.4f}\tAcc: {:.4f} ({:.4f}/{:.4f})'
                string = string.format(e, args.epochs, batch_idx + 1,
                                       args.iterations, args.scheduler.get_last_lr()[0],
                                       train_loss/(batch_idx+1), 100.*correct/total, correct, total)
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
            del(inputs, targets)

        # Update the scheduler
        avg_loss /= len(train_loader)
        if val_loader is not None:
            val_acc, val_loss = val(net, val_loader, criterion_val, device=args.device, supervision='full')
            #val_accuracies.append(val_acc)
            #metric = -val_acc
        else:
            metric = avg_loss

        writer.add_scalar('train/1.train_loss', losses_meter.avg, e)
        writer.add_scalar('test/1.test_acc', val_acc.avg, e)
        writer.add_scalar('test/2.test_loss', val_loss.avg, e)

        # Save the weights
        if e % save_epoch == 0 and args.save == True:
            save_model(net, utils.camel_to_snake(str(net.__class__.__name__)),
                       train_loader.dataset.name, args.save_dir, epoch=args.epochs, metric=abs(-val_acc.avg))


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

def mixup_data(x, y, alpha=1.0, device='cpu'):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Returns the mixup loss based on a certain loss function (criterion)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
