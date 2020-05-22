import visdom
from datasets import get_dataset, HyperX
import utils
import numpy as np
import sklearn.svm
import seaborn as sns
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from torch.nn import init
import torch.utils.data as data
from torchsummary import summary

import math
import os
import datetime
from sklearn.externals import joblib
from tqdm import tqdm

vis = visdom.Visdom()


salinas_img, salinas_gt, salinas_label_values, salinas_ignored_labels, salinas_rgb_bands, salinas_palette = get_dataset("Salinas")

N_CLASSES = len(salinas_label_values)
N_BANDS = salinas_img.shape[-1]

if salinas_palette is None:
    # Generate color palette
    salinas_palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(salinas_label_values) - 1)):
        salinas_palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in salinas_palette.items()}

def convert_to_color(x):
    return utils.convert_to_color_(x, palette=salinas_palette)
def convert_from_color(x):
    return utils.convert_from_color_(x, palette=invert_palette)

SAMPLE_PERCENTAGE = 0.3
SAMPLING_MODE = 'disjoint' #random, fixed, disjoint

train_gt, test_gt = utils.sample_gt(salinas_gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(salinas_gt)))

utils.display_predictions(convert_to_color(train_gt), vis, caption="Train ground truth")
utils.display_predictions(convert_to_color(test_gt), vis, caption="Test ground truth")

device = torch.device('cpu')

hyperparams = {'patch_size' : 1, 'ignored_labels' : salinas_ignored_labels, 'flip_augmentation' : False,
              'radiation_augmentation' : False, 'mixture_augmentation' : False, 'center_pixel' : True,
              'supervision' : 'full', 'batch_size' : 100, 'epochs' : 100, 'dataset' : 'Salinas',
              'n_classes' : N_CLASSES, 'test_stride' : 1, 'scheduler' : None, 'weights' : None,
              'device' : device, 'n_bands' : N_BANDS, 'warmup' : 0, 'threshold' : 0.95, 'labeled' : True}

weights = torch.ones(N_CLASSES)
weights[torch.LongTensor(salinas_ignored_labels)] = 0
weights = weights.to(device)

hyperparams['patch_size'] = 5
hyperparams['center_pixel'] = True
hyperparams['epochs'] = 10
hyperparams['warmup'] = 1
hyperparams['batch_size'] = 64

hyperparams['flip_augmentation'] = True

model = HamidaEtAl(hyperparams['n_bands'], hyperparams['n_classes'],
                   patch_size=hyperparams['patch_size'])
lr =  0.03
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
loss = nn.CrossEntropyLoss(weight=weights)

train_gt, val_gt = utils.sample_gt(train_gt, 0.95, mode='disjoint')

val_dataset = HyperX(salinas_img, val_gt, **hyperparams)
val_loader = data.DataLoader(val_dataset,
                             batch_size=hyperparams['batch_size'])

samples = np.count_nonzero(train_gt)
unlabeled_portion = 7

train_labeled_gt, train_unlabeled_gt = utils.sample_gt(train_gt, 1/(unlabeled_portion + 1), mode='disjoint')

train_labeled_dataset = HyperX(salinas_img, train_labeled_gt, **hyperparams)
train_labeled_loader = data.DataLoader(train_labeled_dataset, batch_size=hyperparams['batch_size'],
                               shuffle=True, drop_last=True)

hyperparams['labeled'] = False
train_unlabeled_dataset = HyperX(salinas_img, train_unlabeled_gt, **hyperparams)
train_unlabeled_loader = data.DataLoader(train_unlabeled_dataset,
                                         batch_size=hyperparams['batch_size']*unlabeled_portion,
                                         shuffle=True, drop_last=True)


amount_labeled = samples//(unlabeled_portion + 1)

iterations = amount_labeled // hyperparams['batch_size']
total_steps = iterations * hyperparams['epochs']
hyperparams['scheduler'] = get_cosine_schedule_with_warmup(optimizer,
                                                           hyperparams['warmup']*iterations, total_steps)

utils.display_predictions(convert_to_color(train_labeled_gt), vis, caption="Labeled train ground truth")
utils.display_predictions(convert_to_color(train_unlabeled_gt), vis, caption="Unlabeled train ground truth")
utils.display_predictions(convert_to_color(val_gt), vis, caption="Validation ground truth")

CLASS_BALANCING = True

if CLASS_BALANCING:
    weights_balance = utils.compute_imf_weights(train_gt, hyperparams['n_classes'], salinas_ignored_labels)
    hyperparams['weights'] = torch.from_numpy(weights_balance)

CHECKPOINT = None #checkpoint to load weights from, string from where to load model

print(hyperparams)
print("Network :")
with torch.no_grad():
    for input, _ in train_labeled_loader:
        break
    summary(model.to(hyperparams['device']), input.size()[1:])
    # We would like to use device=hyperparams['device'] altough we have
    # to wait for torchsummary to be fixed first.

if CHECKPOINT is not None:
    model.load_state_dict(torch.load(CHECKPOINT))

try:
    train(model, optimizer, loss, train_labeled_loader, train_unlabeled_loader, hyperparams['epochs'],
          scheduler=hyperparams['scheduler'], device=hyperparams['device'], threshold=hyperparams['threshold'],
          val_loader=val_loader, display=vis)
except KeyboardInterrupt:
    # Allow the user to stop the training
    pass

probabilities = test(model, salinas_img, hyperparams)
prediction = np.argmax(probabilities, axis=-1)

run_results = utils.metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])

mask = np.zeros(salinas_gt.shape, dtype='bool')
for l in hyperparams['ignored_labels']:
    mask[salinas_gt == l] = True
prediction[mask] = 0

color_prediction = convert_to_color(prediction)
utils.display_predictions(color_prediction, vis, gt=convert_to_color(test_gt), caption="Prediction vs. test ground truth")

utils.show_results(run_results, vis, label_values=salinas_label_values)


def train(net, optimizer, criterion, labeled_data_loader, unlabeled_data_loader, epoch, threshold, scheduler=None,
          display_iter=100, device=torch.device('cpu'), display=None,
          val_loader=None):
    """
    Training loop to optimize a network for several epochs and a specified loss
    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        labeled_data_loader: a PyTorch dataset loader for the labeled dataset
        unlabeled_data_loader: a PyTorch dataset loader for the weakly and strongly augmented, unlabeled dataset
        epoch: int specifying the number of training epochs
        threshold: probability thresold for pseudo labels acceptance
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

        train_loader = zip(labeled_data_loader, unlabeled_data_loader)

        # Run the training loop for one epoch
        for batch_idx, (data_x, data_u) in tqdm(enumerate(train_loader), total=len(labeled_data_loader)):
            # Load the data into the GPU if required
            inputs_x, targets_x = data_x
            inputs_u_w, inputs_u_s = data_u

            batch_size = inputs_x.shape[0]

            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(device)
            targets_x = targets_x.to(device)
            logits = net(inputs)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            del logits

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(threshold).float()

            Lu = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()

            loss = Lx + 1 * Lu


            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(e, epoch, batch_idx * len(data_x), len(data_x) * len(labeled_data_loader),
                                       100. * batch_idx / len(labeled_data_loader), mean_losses[iter_])
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
            del(data_x, data_u, loss)

        # Update the scheduler
        avg_loss /= len(labeled_data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision='full')
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(net, utils.camel_to_snake(str(net.__class__.__name__)),
                       labeled_data_loader.dataset.name, epoch=e, metric=abs(metric))


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


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
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

def val(net, data_loader, device='cpu', supervision='full'):
    """
    Validate the model on the validation dataset
    """
    # TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
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
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total


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

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
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
