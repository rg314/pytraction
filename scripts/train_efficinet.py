import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pytraction.net.utlis import get_img_from_seg, visualize
from pytraction.net.dataloader import Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing
from pytraction.net.preproc import CellTrackPreprocessor

import matplotlib.pyplot as plt 
import torch
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from collections import defaultdict

# need to update path
DATA_DIR = 'data/..'
PATH = '/home/ryan/Documents/GitHub/pytraction/data'

DATASETS = ['DIC-C2DH-HeLa','Fluo-C2DL-Huh7', 'Fluo-C2DL-MSC', 'Fluo-N2DH-GOWT1','PhC-C2DH-U373',]
ENCODER = 'efficientnet-b1'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['cell']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'
SIZE=512
IN_CHANNELS = 1
VIS = False
TRAIN = True
BATCH_SIZE = 4

ctp = CellTrackPreprocessor(PATH, DATASETS, gt_type='SEG', gt_standard='GT')

x_images = ctp.imgs
y_images = ctp.masks


assert y_images != [], f'No images found. Please check format of {VALID_DATASETS}'	
	

x_train, x_valid, y_train, y_valid = train_test_split(x_images, y_images, test_size=0.20, random_state=42)

if VIS:
    
    train_dataset = Dataset(
        x_train,
        y_train,
        augmentation=get_training_augmentation(size=SIZE), 
        # preprocessing=get_preprocessing(preprocessing_fn),``
        classes=CLASSES,
    )

    for i in range(100):
        image, mask = train_dataset[i]
        visualize(image=image, mask=mask)
        plt.savefig('test.png')




# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels=IN_CHANNELS,
)


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = Dataset(
    x_train,
	y_train,
    augmentation=get_training_augmentation(size=SIZE), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    in_channels=IN_CHANNELS,
)

valid_dataset = Dataset(
    x_valid,
	y_valid,
    augmentation=get_validation_augmentation(size=SIZE), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
    in_channels=IN_CHANNELS,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)



if TRAIN:

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    loss = smp.utils.losses.JaccardLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])


    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )



    # train model for 40 epochs

    max_score = 0
    
    logs = {
        'train': {
            'iou_score': [],
            'jaccard_loss': [],

        },
        'valid': {
            'iou_score': [],
            'jaccard_loss': [],

        },
        'epoch': []
    }

    fig, ax = plt.subplots(1,2)
    
    ax[0].plot(logs['epoch'], logs['train']['iou_score'], label='train', color='blue')
    ax[0].plot(logs['epoch'], logs['valid']['iou_score'], label='valid', color='orange')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('IoU score')
    ax[0].legend()

    ax[1].set_title('Jaccard loss')
    ax[1].plot(logs['epoch'], logs['train']['jaccard_loss'], label='train', color='blue')
    ax[1].plot(logs['epoch'], logs['valid']['jaccard_loss'], label='valid', color='orange')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Jaccard loss')
    ax[1].legend()

    for i in range(0, 40):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, 'best_model_1.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        logs['train']['iou_score'].append(train_logs['iou_score'])
        logs['valid']['iou_score'].append(valid_logs['iou_score'])
        logs['train']['jaccard_loss'].append(train_logs['jaccard_loss'])
        logs['valid']['jaccard_loss'].append(valid_logs['jaccard_loss'])
        logs['epoch'].append(i)

        plt.suptitle(ENCODER)

        ax[0].set_title('IoU score')
        ax[0].plot(logs['epoch'], logs['train']['iou_score'], label='train', color='blue')
        ax[0].plot(logs['epoch'], logs['valid']['iou_score'], label='valid', color='orange')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('IoU score')

        ax[1].set_title('Jaccard loss')
        ax[1].plot(logs['epoch'], logs['train']['jaccard_loss'], label='train', color='blue')
        ax[1].plot(logs['epoch'], logs['valid']['jaccard_loss'], label='valid', color='orange')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Jaccard loss')

        plt.tight_layout()
        plt.pause(0.5)
        # fig.canvas.draw()


    plt.show()
