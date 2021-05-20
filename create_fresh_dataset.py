# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import GPUtil
import os
print("My working directory is: " , os.getcwd())


from models.matcher import HungarianMatcher
from models.detr import SetCriterion

import numpy as np
import pandas as pd
from datetime import datetime
import time
import random

from tqdm import tqdm

#sklearn
from sklearn.model_selection import StratifiedKFold

#CV
import cv2

#matplotlib
import matplotlib.pyplot as plt

#albumentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

#Glob
from glob import glob

### UTILS - AverageMeter
### class for averaging loss, metric, etc over epochs

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

### CONFIGURATION -- Basic configuration for this model
n_folds = 5
seed = 42
num_classes = 2
num_queries = 100
null_class_coef = 0.5
BATCH_SIZE = 4 #8
LR = 2e-5
EPOCHS = 1 #2

### SEED - for reproducible results
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed)

### DATA PREP
marking = pd.read_csv('C:\\Users\\Eva.Locusteanu\\PycharmProjects\\detr\\models\\train.csv')

bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    marking[column] = bboxs[:,i]
marking.drop(columns=['bbox'], inplace=True)

# Creating Folds
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

df_folds = marking[['image_id']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

df_folds.to_csv("DFfolds.csv")
print(n_folds) #5
print(fold_number) #4 #what is this?
print(df_folds.head()) #why do I not just export this?

#check GPU utilization
GPUtil.showUtilization()