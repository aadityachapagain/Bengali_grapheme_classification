import pandas as pd
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from matplotlib import pyplot as plt
from torch.utils import data
import copy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--classifier", help="vowel_diacritic, consonant_diacritic, grapheme_root")
parser.add_argument("--fineTune", help="yes or no", default="yes")
parser.add_argument("--pretrained", help="yes or no", default="yes")
parser.add_argument("--freeze", help="yes or no", default= "no")
parser.add_argument("--save", help= "yes or no", default="yes")
parser.add_argument("--overfit", help= "yes or no", default="no")
parser.add_argument("--mode", help= "train or test", default="train")
parser.add_argument("--epoch", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001, help = "learning rate")
parser.add_argument("--batch_size", type= int, default=32, help= "Batch size for training")
args = parser.parse_args()

batch_size = args.batch_size


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_files = []
for file in os.listdir('data'):
    if file.startswith('train_'):
        data_files.append(pd.read_parquet(os.path.join('data',file)))
#         concat the dataframe
train_dataset = pd.concat(data_files)


data_files = []
for file in os.listdir('data'):
    if file.startswith('test_'):
        data_files.append(pd.read_parquet(os.path.join('data',file)))
#         concat the dataframe
test_dataset = pd.concat(data_files)

# Preprocess image
tfms = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Bengali_Grapheme_Data_Loader(data.Dataset):
    
    def __init__(self, data_split ,data_path, classifier, transforms = None):
        """
        @data_split: data_loader for either train or test
        @data_path: train and test data path
        @classifer: name of classifier you want to load data for
        """
        self.transforms = transforms
        self.data_split = data_split
        self.data_path = data_path
        self.classifier = classifier
        if data_split.lower() not in ['train','test']:
            raise ValueError('data_split must be either train or test')
        
        self.pxls = test_dataset.columns[test_dataset.columns != 'image_id']
        
        if not len(data_files):
            raise ValueError(f'No any {data_split} data found in data path !')
        
        self.label = pd.read_csv(f'{data_path}/{data_split}.csv')
        if data_split.lower() == 'train':
            if classifier not in ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']:
                raise ValueError('Wrong classifier, classifier means each domain of prediction')
        else:
            if classifier not in self.label['component'].unique():
                raise ValueError('Wrong classifier, classifier means each domain of prediction')
        
            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ids = self.label.iloc[index]['image_id']
        if self.data_split.lower() == 'train':
            # Load data and get label
            idx_img = train_dataset[train_dataset['image_id'] == ids].index[0]
            # convert objecttype numpy array to float to display in imshow
            X = train_dataset.iloc[idx_img][self.pxls].values.reshape(137, 236).astype(np.uint8)
        else:
            idx_img = test_dataset[test_dataset['image_id'] == ids].index[0]
            # convert objecttype numpy array to float to display in imshow
            X = test_dataset.iloc[idx_img][self.pxls].values.reshape(137, 236).astype(np.uint8)
        X = np.repeat(X[..., np.newaxis], 3, -1)
        if self.transforms:
            X = self.transforms(X)
        else:
            X = torch.from_numpy(X).double()
        if self.data_split.lower()  == 'test':
                return X, ids
        y = self.label.iloc[index][self.classifier]
        return X, y
    
    def  get_classes(self):
        'return no of classes for each classifier'
        classes_df = pd.read_csv(f'{self.data_path}/class_map.csv')
        return classes_df[classes_df['component_type'] == self.classifier]['label'].unique()

def train_model(model, dataloader,criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += float(torch.sum(preds == labels.data))/float(len(preds))
        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_corrects / len(dataloader)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            if args.save == 'yes':
                torch.save(model.state_dict(), f'../gdrive/My Drive/kaggle_bengali/{args.classifier}__model__ResNext')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

train_data_loader = Bengali_Grapheme_Data_Loader(args.mode, 'data',args.classifier, tfms)
dataloader = torch.utils.data.DataLoader(train_data_loader, batch_size=batch_size, shuffle=True, num_workers= 4)

if args.pretrained == 'yes':
    model_ft = torchvision.models.resnext101_32x8d(pretrained=True)
else:
    model_ft = torchvision.models.resnext101_32x8d(pretrained=False)

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(train_data_loader.get_classes()))

if os.path.isfile(f'../gdrive/My Drive/kaggle_bengali/{args.classifier}__model__ResNext'):
    model_ft.load_state_dict(torch.load(f'../gdrive/My Drive/kaggle_bengali/{args.classifier}__model__ResNext'))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

if args.overfit == 'yes':
    print( "Lets Overfit !")
    overfit_data = [next(iter(dataloader))]
    args.save = 'no'
    model_ft = train_model(model_ft, overfit_data, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epoch)
else:
    print('Lets Train !')
    model_ft = train_model(model_ft, dataloader, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epoch)

torch.save(model_ft.state_dict(), f'../gdrive/My Drive/kaggle_bengali/{args.classifier}__model__ResNext')