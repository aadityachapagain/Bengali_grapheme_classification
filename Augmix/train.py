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

from model import accuracy, build_classifier

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

parser = argparse.ArgumentParser(description='Trains an SeResnext-101 32dx8')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=90, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.004,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=16, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=32)
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
    default=.95,
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

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datadir = Path('data')
featherdir = Path('data')
outdir = Path('.')

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
              example = self.transform(example[0]), example[1].astype(np.int64)
            else:
              example = self.transform(example)
        else:
          if self.transform:
            if self.train:
              examples = example[0]
              examples = tuple([self.transform(exm) for exm in examples])
              y = example[1]
              example = tuple([examples, y.astype(np.int64)])
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

        return x

def prepare_image(datadir, featherdir, data_type='train',
                  submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
                         for i in indices]
    else:
        image_df_list = [pd.read_feather(featherdir / f'{data_type}_image_data_{i}.feather')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images

def train_model(model, dataloaders, optimizer, scheduler, num_epochs=17):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_stats = {
            'loss': 0,
            'loss_grapheme': 0,
            'loss_vowel': 0,
            'loss_consonant': 0,
            'acc1_grapheme': 0, 'acc5_grapheme': 0,
            'acc1_vowel': 0, 'acc5_vowel':0,
            'acc1_consonant': 0, 'acc5_consonant':0
        }

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if phase == 'train':
                  pass
                else:
                  inputs = inputs.to(device)
              
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    phase_logit = phase == 'train'
                    loss,metrics = model(inputs, labels, no_jsd = not phase_logit)
                    # loss, _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_stats['loss_grapheme'] += metrics['loss_grapheme']
                running_stats['loss_vowel'] += metrics['loss_vowel']
                running_stats['loss_consonant'] += metrics['loss_consonant']
                
                running_stats['acc1_grapheme'] += metrics['acc1_grapheme']
                running_stats['acc1_vowel'] += metrics['acc1_vowel']
                running_stats['acc1_consonant'] += metrics['acc1_consonant']
                
                running_stats['acc5_grapheme'] += metrics['acc5_grapheme']
                running_stats['acc5_vowel'] += metrics['acc5_vowel']
                running_stats['acc5_consonant'] += metrics['acc5_consonant']

            print('-' * 10)
            print(f'phase: {phase}')
            print(f'total loss: {(running_stats["loss_grapheme"] + running_stats["loss_vowel"] + running_stats["loss_consonant"])/len(dataloaders[phase])}')
            print(f'grapheme top-1 acc: {running_stats["acc1_grapheme"]/len(dataloaders[phase])}, vowel acc: {running_stats["acc1_vowel"]/len(dataloaders[phase])},consonent acc:{running_stats["acc1_consonant"]/len(dataloaders[phase])}')
            print(f'grapheme top-k acc: {running_stats["acc5_grapheme"]/len(dataloaders[phase])}, vowel acc: {running_stats["acc5_vowel"]/len(dataloaders[phase])},consonent acc:{running_stats["acc5_consonant"]/len(dataloaders[phase])}')

            epoch_acc = (running_stats['acc1_grapheme'] + running_stats['acc5_grapheme'])/2
            # deep copy the model
            if phase == 'val' and  epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                torch.save(model.state_dict(), f'../gdrive/My Drive/bengali_ghrapheme/predictor_3.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# --- Model ---
device = torch.device(device)
n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant
print('n_total', n_total)

classifier = build_classifier(arch = 'pretrained', load_model_path= None, n_total = n_total, device = device)
classifier.load_state_dict(torch.load('../gdrive/My Drive/bengali_ghrapheme/predictor_3.pt'))

# create labels
labels = pd.read_csv('data/train.csv')
labels = labels[labels.columns[1:4]]
labels = labels.values

train_images_raw = prepare_image(datadir, datadir, 'train', submission = True)

# Train test split

msk = np.random.rand(len(train_images_raw)) > 0.009

train_image_split = train_images_raw[msk]
train_labels = labels[msk]

valid_image_split = train_images_raw[~msk]
val_labels = labels[~msk]
print('train images len: {}, val images len: {}'.format(len(train_image_split), len(valid_image_split)))
print('train images len: {}, val images len: {}'.format(len(train_labels), len(val_labels)))

# data loader

train_dataset = AugMixDataset(train_image_split)

train_dataset = BengaliAIDataset(
    train_dataset, train_labels,
    transform=Transform(affine=False, crop=True, size=(augmentations.IMAGE_SIZE, augmentations.IMAGE_SIZE),
                        threshold=20, train=True))
print('train_dataset', len(train_dataset))

val_dataset = BengaliAIDataset(
    valid_image_split, val_labels,
    transform=Transform(affine=False, crop=True, size=(augmentations.IMAGE_SIZE, augmentations.IMAGE_SIZE),
                        threshold=20, train=True))
print('val_dataset', len(train_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

print('train iterations: {}, val iterations: {}'.format(len(train_loader), len(val_loader)))

data_loaders = {'train':train_loader, 'val': val_loader}
optimizer_ft = torch.optim.SGD(
    classifier.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.decay)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=9, gamma=0.1)
trained_model = train_model(classifier, data_loaders, optimizer_ft, exp_lr_scheduler)
torch.save(trained_model.state_dict(), f'../gdrive/My Drive/bengali_ghrapheme/predictor_3.pt')