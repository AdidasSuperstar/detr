# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import GPUtil
import os

print("My working directory is: ", os.getcwd())

from models.matcher import HungarianMatcher
from models.detr import SetCriterion

import numpy as np
import pandas as pd
from datetime import datetime
import time
import random

from tqdm import tqdm

# sklearn
from sklearn.model_selection import StratifiedKFold

# CV
import cv2

# matplotlib
import matplotlib.pyplot as plt

# albumentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Glob
from glob import glob

torch.cuda.empty_cache()
GPUtil.showUtilization()

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
n_folds = 3
seed = 42
num_classes = 2
num_queries = 64
null_class_coef = 0.5
BATCH_SIZE = 1  # 8
LR = 2e-5
EPOCHS = 2  # 2


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
marking = pd.read_csv('C:\\Users\\Eva.Locusteanu\\PycharmProjects\\detr\\Point_ONE_Percent_MiniTrainingData.csv')

bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    marking[column] = bboxs[:, i]
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

# print(n_folds) #5
# print(fold_number) #4 #what is this? Really?? Just a column in the bloody dataset??!
# print(df_folds.head())

# check GPU utilization
GPUtil.showUtilization()


###AUGMENTATIONS
def get_train_transforms():
    return A.Compose(
        [A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),

                  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)], p=0.9),

         A.ToGray(p=0.01),

         A.HorizontalFlip(p=0.5),

         A.VerticalFlip(p=0.5),

         A.Resize(height=512, width=512, p=1),

         A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),

         ToTensorV2(p=1.0)],

        p=1.0,

        bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
    )


def get_valid_transforms():
    return A.Compose([A.Resize(height=512, width=512, p=1.0),
                      ToTensorV2(p=1.0)],
                     p=1.0,
                     bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0, label_fields=['labels'])
                     )


print("So far so good")
###CREATE DATASET
DIR_TRAIN = "C:\\Users\\Eva.Locusteanu\\PycharmProjects\\detr\\models\\train_subset_3"  # sample of about 300 imgs (instead of total 3423 images)


class WheatDataset(Dataset):
    def __init__(self, image_ids, dataframe, transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.transforms = transforms

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{DIR_TRAIN}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # DETR takes in data in coco format
        boxes = records[['x', 'y', 'w', 'h']].values

        # Area of bb
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # AS pointed out by PRVI It works better if the main class is labelled as zero
        labels = np.zeros(len(boxes), dtype=np.int32)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

        # Normalizing BBOXES

        _, h, w = image.shape
        boxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'], rows=h, cols=w)
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area

        return image, target, image_id


print("So far so good  - part 2")

# check GPU utilization
GPUtil.showUtilization()


###MODEL
class DETRModel(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(DETRModel, self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features

        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries

    def forward(self, images):
        return self.model(images)


###Matcher and Bipartite Matching Loss
'''
code taken from github repo detr , 'code present in engine.py'
'''

matcher = HungarianMatcher()

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou': 1}

losses = ['labels', 'boxes', 'cardinality']


### Training Function
def train_fn(data_loader, model, criterion, optimizer, device, scheduler, epoch):
    model.train()
    criterion.train()

    summary_loss = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for step, (images, targets, image_ids) in enumerate(tk0):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        output = model(images)

        loss_dict = criterion(output, targets)
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()

        losses.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        summary_loss.update(losses.item(), BATCH_SIZE)
        tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


### Evaluation Function
def eval_fn(data_loader, model, criterion, device):
    model.eval()
    criterion.eval()
    summary_loss = AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for step, (images, targets, image_ids) in enumerate(tk0):
            losses = method_name_test(criterion, device, images, model, targets)

            summary_loss.update(losses.item(), BATCH_SIZE)
            tk0.set_postfix(loss=summary_loss.avg)

    return summary_loss


def method_name_test(criterion, device, images, model, targets):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    output = model(images)
    loss_dict = criterion(output, targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    return losses


### Engine
def collate_fn(batch):
    return tuple(zip(*batch))


# check GPU utilization
GPUtil.showUtilization()


def run(fold):
    df_train = df_folds[df_folds['fold'] != fold]
    df_valid = df_folds[df_folds['fold'] == fold]

    train_dataset = WheatDataset(
        image_ids=df_train.index.values,
        dataframe=marking,
        transforms=get_train_transforms()
    )

    valid_dataset = WheatDataset(
        image_ids=df_valid.index.values,
        dataframe=marking,
        transforms=get_valid_transforms()
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1, #someone online says 0? or not 0? not sure
        collate_fn=collate_fn
    )

    device = torch.device('cuda') #should be the same as 'cuda:0' or 'cuda'
    model = DETRModel(num_classes=num_classes, num_queries=num_queries)
    model = model.to(device)
    criterion = SetCriterion(num_classes - 1, matcher, weight_dict, eos_coef=null_class_coef, losses=losses)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_loss = 10 ** 5
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_data_loader, model, criterion, optimizer, device, scheduler=None, epoch=epoch)
        valid_loss = eval_fn(valid_data_loader, model, criterion, device)

        print('|EPOCH {}| TRAIN_LOSS {}| VALID_LOSS {}|'.format(epoch + 1, train_loss.avg, valid_loss.avg))

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            print('Best model found for Fold {} in Epoch {}........Saving Model'.format(fold, epoch + 1))
            torch.save(model.state_dict(), f'detr_best_{fold}.pth')



GPUtil.showUtilization()

'''
if __name__ == '__main__':
    model = run(fold=0)
    model.cuda()
'''
# check GPU utilization
GPUtil.showUtilization()

torch.cuda.memory_summary(device=None, abbreviated=False)


def view_sample(df_valid, model, device):
    '''
    Code taken from Peter's Kernel
    https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
    '''
    valid_dataset = WheatDataset(image_ids=df_valid.index.values,
                                 dataframe=marking,
                                 transforms=get_valid_transforms()
                                 )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn)

    images, targets, image_ids = next(iter(valid_data_loader))
    _, h, w = images[0].shape  # for de normalizing images

    images = list(img.to(device) for img in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    boxes = targets[0]['boxes'].cpu().numpy()
    boxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(boxes, h, w)]
    sample = images[0].permute(1, 2, 0).cpu().numpy()

    model.eval()
    model.to(device)
    cpu_device = torch.device("cpu")

    with torch.no_grad():
        outputs = model(images)

    outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2] + box[0], box[3] + box[1]),
                      (220, 0, 0), 1)

    oboxes = outputs[0]['pred_boxes'][0].detach().cpu().numpy()
    oboxes = [np.array(box).astype(np.int32) for box in A.augmentations.bbox_utils.denormalize_bboxes(oboxes, h, w)]
    prob = outputs[0]['pred_logits'][0].softmax(1).detach().cpu().numpy()[:, 0]

    for box, p in zip(oboxes, prob):

        if p > 0.5:
            color = (0, 0, 220)  # if p>0.5 else (0,0,0)
            cv2.rectangle(sample,
                          (box[0], box[1]),
                          (box[2] + box[0], box[3] + box[1]),
                          color, 1)

    ax.set_axis_off()
    ax.imshow(sample)


if __name__ == '__main__':
    model = DETRModel(num_classes=num_classes, num_queries=num_queries)
    model.load_state_dict(torch.load("./detr_best_0.pth"))
    abc = view_sample(df_folds[df_folds['fold'] == 0], model=model, device=torch.device('cuda'))
    abc.cuda()
    # model = run(fold=0)
    # model.cuda()

