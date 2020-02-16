from __future__ import print_function

import argparse
import os
import shutil
import time

import augmentations

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
import six
from torch import nn
import math
import torch.optim as optim
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import Dataset
from torch.optim import lr_scheduler
from torch.nn import Sequential

import gc
import os
from pathlib import Path
import random
import sys
from typing import List
import time
import copy

from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from skimage.transform import AffineTransform, warp

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# import cumix and mixup augmentation
from augmentation import cutmix, cutmix_criterion, mixup, mixup_criterion, ohem_loss

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb

import pretrainedmodels


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__') and
                     callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains an SeResnext-101 32dx8')
parser.add_argument(
    'clean_data', metavar='DIR', help='path to clean ImageNet dataset')
parser.add_argument(
    'corrupted_data', metavar='DIR_C', help='path to ImageNet-C dataset')
parser.add_argument(
    '--model',
    '-m',
    default='resnet50',
    choices=model_names,
    help='model architecture: ' + ' | '.join(model_names) +
    ' (default: resnet50)')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=90, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.02,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=256, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=64)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0001,
    help='Weight decay (L2 penalty).')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=1,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--aug-prob-coeff',
    default=1.,
    type=float,
    help='Probability distribution coefficients')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=10,
    help='Training loss print frequency (batches).')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

# Raw AlexNet errors taken from https://github.com/hendrycks/robustness
ALEXNET_ERR = [
    0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
    0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
    0.606500
]


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR (linearly scaled to batch size) decayed by 10 every n / 3 epochs."""
  b = args.batch_size / 256.
  k = args.epochs // 3
  if epoch < k:
    m = 1
  elif epoch < 2 * k:
    m = 0.1
  else:
    m = 0.01
  lr = args.learning_rate * m * b
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k."""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_mce(corruption_accs):
  """Compute mCE (mean Corruption Error) normalized by AlexNet performance."""
  mce = 0.
  for i in range(len(CORRUPTIONS)):
    avg_err = 1 - np.mean(corruption_accs[CORRUPTIONS[i]])
    ce = 100 * avg_err / ALEXNET_ERR[i]
    mce += ce / 15
  return mce


def aug(image):
  """Perform AugMix augmentations and compute mixture.
  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.
  Returns:
    mixed: Augmented and mixed image.
  """
  preprocess = transforms.ToTensor()
  image = transforms.ToPILImage()(image)
  image = image.resize((augmentations.IMAGE_SIZE,augmentations.IMAGE_SIZE))
  aug_list = augmentations.augmentations
  if args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(
      np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
  m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

  mix = torch.zeros_like(preprocess(image))
  for i in range(args.mixture_width):
    image_aug = image.copy()
    depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  mixed = (np.array(mixed)* 255).astype(np.uint8)
  return mixed[0]

class DatasetMixin(torch.utils.data.Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __getitem__(self, index):
        """Returns an example or a sequence of examples."""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example_wrapper(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return [self.get_example_wrapper(i) for i in index]
        else:
            return self.get_example_wrapper(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example_wrapper(self, i):
        """Wrapper of `get_example`, to apply `transform` if necessary"""
        example = self.get_example(i)
        if self.no_jsd:
          if self.transform:
            if self.train:
              example = self.transform(example[0]), example[1]
            else:
              example = self.transform(example)
        else:
          if self.transform:
            if self.train:
              examples = example[0]
              examples = tuple([self.transform(exm) for exm in examples])
              y = example[1]
              example = tuple([examples, y])
            else:
              example = tuple([self.transform(exm) for exm in example])
        return example

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError


class BengaliAIDataset(DatasetMixin):
    def __init__(self, images, labels=None, transform=None, indices=None):
        super(BengaliAIDataset, self).__init__(transform=transform)
        self.images = images
        if isinstance(images ,AugMixDataset):
          self.no_jsd = images.no_jsd
        else:
          self.no_jsd = True
        self.labels = labels
        if indices is None:
            indices = np.arange(len(images))
        self.indices = indices
        self.train = labels is not None

    def __len__(self):
        """return length of this dataset"""
        return len(self.indices)

    def get_example(self, i):
        """Return i-th data"""
        i = self.indices[i]
        if not self.no_jsd:
          x = self.images[i]
          x = tuple([(255 - xim).astype(np.float32) / 255. for xim in x])
        else:
          x = self.images[i]
          x = (255 - x).astype(np.float32) / 255.
        # Opposite white and black: background will be white (1.0) and
        # for future Affine transformation
        
        if self.train:
            y = self.labels[i]
            return x, y
        else:
            return x

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, no_jsd=False):
    self.dataset = dataset
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x = self.dataset[i]
    if self.no_jsd:
      return aug(x)
    else:
      im_tuple = (x, aug(x), aug(x))
      return im_tuple

  def __len__(self):
    return len(self.dataset)

def affine_image(img):
    """

    Args:
        img: (h, w) or (1, h, w)

    Returns:
        img: (h, w)
    """
    # ch, h, w = img.shape
    # img = img / 255.
    if img.ndim == 3:
        img = img[0]

    # --- scale ---
    min_scale = 0.8
    max_scale = 1.2
    sx = np.random.uniform(min_scale, max_scale)
    sy = np.random.uniform(min_scale, max_scale)

    # --- rotation ---
    max_rot_angle = 12
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle) * np.pi / 180.

    # --- shear ---
    max_shear_angle = 20
    shear_angle = np.random.uniform(-max_shear_angle, max_shear_angle) * np.pi / 180.

    # --- translation ---
    max_translation = 4
    tx = np.random.randint(-max_translation, max_translation)
    ty = np.random.randint(-max_translation, max_translation)

    tform = AffineTransform(scale=(sx, sy), rotation=rot_angle, shear=shear_angle,
                            translation=(tx, ty))
    transformed_image = warp(img, tform)
    assert transformed_image.ndim == 2
    return transformed_image


def crop_char_image(image, threshold=40./255.):
    assert image.ndim == 2
    is_black = image > threshold

    is_black_vertical = np.sum(is_black, axis=0) > 0
    is_black_horizontal = np.sum(is_black, axis=1) > 0
    left = np.argmax(is_black_horizontal)
    right = np.argmax(is_black_horizontal[::-1])
    top = np.argmax(is_black_vertical)
    bottom = np.argmax(is_black_vertical[::-1])
    height, width = image.shape
    cropped_image = image[left:height - right, top:width - bottom]
    return cropped_image


def resize(image, size=(128, 128)):
    return cv2.resize(image, size)

def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0., 1.)
    return x


class Transform:
    def __init__(self, affine=True, crop=True, size=(64, 64),
                 normalize=True, train=True, threshold=40.,
                 sigma=-1.):
        self.affine = affine
        self.crop = crop
        self.size = size
        self.normalize = normalize
        self.train = train
        self.threshold = threshold / 255.
        self.sigma = sigma / 255.

    def __call__(self, example):
        if self.train:
            x, y = example
        else:
            x = example
        # --- Augmentation ---
        if self.affine:
            x = affine_image(x)

        # --- Train/Test common preprocessing ---
        if self.crop:
            x = crop_char_image(x, threshold=self.threshold)
        if self.size is not None:
            x = resize(x, size=self.size)
        if self.sigma > 0.:
            x = add_gaussian_noise(x, sigma=self.sigma)
        if self.normalize:
            x = (x.astype(np.float32) - 0.0692) / 0.2051
        if x.ndim == 2:
            x = x[None, :, :]
        x = x.astype(np.float32)
        if self.train:
            y = y.astype(np.int64)
            return x, y
        else:
            return x

def train(net, train_loader, optimizer):
  """Train for one epoch."""
  net.train()
  data_ema = 0.
  batch_ema = 0.
  loss_ema = 0.
  acc1_ema = 0.
  acc5_ema = 0.

  end = time.time()
  for i, (images, targets) in enumerate(train_loader):
    # Compute data loading time
    data_time = time.time() - end
    optimizer.zero_grad()

    if args.no_jsd:
      images = images.cuda()
      targets = targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
    else:
      images_all = torch.cat(images, 0).cuda()
      targets = targets.cuda()
      logits_all = net(images_all)
      logits_clean, logits_aug1, logits_aug2 = torch.split(
          logits_all, images[0].size(0))

      # Cross-entropy is only computed on clean images
      loss = F.cross_entropy(logits_clean, targets)

      p_clean, p_aug1, p_aug2 = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)

      # Clamp mixture distribution to avoid exploding KL divergence
      p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
      loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
      acc1, acc5 = accuracy(logits_clean, targets, topk=(1, 5))  # pylint: disable=unbalanced-tuple-unpacking

    loss.backward()
    optimizer.step()

    # Compute batch computation time and update moving averages.
    batch_time = time.time() - end
    end = time.time()

    data_ema = data_ema * 0.1 + float(data_time) * 0.9
    batch_ema = batch_ema * 0.1 + float(batch_time) * 0.9
    loss_ema = loss_ema * 0.1 + float(loss) * 0.9
    acc1_ema = acc1_ema * 0.1 + float(acc1) * 0.9
    acc5_ema = acc5_ema * 0.1 + float(acc5) * 0.9

    if i % args.print_freq == 0:
      print(
          'Batch {}/{}: Data Time {:.3f} | Batch Time {:.3f} | Train Loss {:.3f} | Train Acc1 '
          '{:.3f} | Train Acc5 {:.3f}'.format(i, len(train_loader), data_ema,
                                              batch_ema, loss_ema, acc1_ema,
                                              acc5_ema))

  return loss_ema, acc1_ema, batch_ema


def test(net, test_loader):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader.dataset), total_correct / len(
      test_loader.dataset)


def test_c(net, test_transform):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = {}
  for c in CORRUPTIONS:
    print(c)
    for s in range(1, 6):
      valdir = os.path.join(args.corrupted_data, c, str(s))
      val_loader = torch.utils.data.DataLoader(
          datasets.ImageFolder(valdir, test_transform),
          batch_size=args.eval_batch_size,
          shuffle=False,
          num_workers=args.num_workers,
          pin_memory=True)

      loss, acc1 = test(net, val_loader)
      if c in corruption_accs:
        corruption_accs[c].append(acc1)
      else:
        corruption_accs[c] = [acc1]

      print('\ts={}: Test Loss {:.3f} | Test Acc1 {:.3f}'.format(
          s, loss, 100. * acc1))

  return corruption_accs


def main():


  if args.pretrained:
    print("=> using pre-trained model '{}'".format(args.model))
    net = models.__dict__[args.model](pretrained=True)
  else:
    print("=> creating model '{}'".format(args.model))
    net = models.__dict__[args.model]()

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  start_epoch = 0

  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc1 = checkpoint['best_acc1']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)

  if args.evaluate:
    test_loss, test_acc1 = test(net, val_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Acc1 {:.3f}'.format(
        test_loss, 100 * test_acc1))

    corruption_accs = test_c(net, test_transform)
    for c in CORRUPTIONS:
      print('\t'.join([c] + map(str, corruption_accs[c])))

    print('mCE (normalized by AlexNet): ', compute_mce(corruption_accs))
    return

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          'imagenet_{}_training_log.csv'.format(args.model))
  with open(log_path, 'w') as f:
    f.write(
        'epoch,batch_time,train_loss,train_acc1(%),test_loss,test_acc1(%)\n')

  best_acc1 = 0
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)

    train_loss_ema, train_acc1_ema, batch_ema = train(net, train_loader,
                                                      optimizer)
    test_loss, test_acc1 = test(net, val_loader)

    is_best = test_acc1 > best_acc1
    best_acc1 = max(test_acc1, best_acc1)
    checkpoint = {
        'epoch': epoch,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%0.3f,%0.6f,%0.2f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          batch_ema,
          train_loss_ema,
          100. * train_acc1_ema,
          test_loss,
          100. * test_acc1,
      ))

    print(
        'Epoch {:3d} | Train Loss {:.4f} | Test Loss {:.3f} | Test Acc1 '
        '{:.2f}'
        .format((epoch + 1), train_loss_ema, test_loss, 100. * test_acc1))

    corruption_accs = test_c(net, test_transform)
    for c in CORRUPTIONS:
      print('\t'.join(map(str, [c] + corruption_accs[c])))

    print('mCE (normalized by AlexNet):', compute_mce(corruption_accs))


if __name__ == '__main__':
  main()