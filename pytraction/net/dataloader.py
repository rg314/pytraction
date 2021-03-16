import cv2
from PIL import Image
import numpy as np
import albumentations as albu

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations. 
    Adopted for CamVid example
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background', 'cell', 'balls']
    
    def __init__(
            self, 
            image_dir,
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            in_channels=3,
            size=256,
            test=False
    ):
        self.ids = len(masks_dir)
        self.masks_fps = masks_dir
        self.images_fps = image_dir
        self.in_channels = in_channels
        self.test = test
        self.size = size
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = Image.open(self.images_fps[i])
        mask = Image.open(self.masks_fps[i])


        # read data
        image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2RGB)
        image = np.asarray(image)[:,:,:3]

        if not self.test:
            mask = np.asarray(mask)
            mask = 1.0 * (mask > 0)
            mask = mask.reshape(mask.shape[0], mask.shape[1],1 )
        
        else:
            mask = np.zeros((self.size, self.size, 1))

        # cv2.imwrite('test.png', mask)
        # image = image.reshape(image.shape + (1,))
        

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask) 
            image, mask = sample['image'], sample['mask']

        if self.in_channels != 3:
            image = image[:1,:,:]
            
        return image, mask
        
    def __len__(self):
        return self.ids


def get_training_augmentation(size):
    SIZE = size
    train_transform = [
        albu.Resize(SIZE,SIZE),

        albu.HorizontalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Rotate(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.IAAPerspective(p=0.5),



        # albu.PadIfNeeded(min_height=SIZE, min_width=SIZE, always_apply=True, border_mode=0),

        # albu.IAAAdditiveGaussianNoise(p=0.2),


    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(size):
    """Add paddings to make image shape divisible by 32"""
    SIZE = size
    test_transform = [
        albu.Resize(SIZE,SIZE),
    ]
    return albu.Compose(test_transform)

def get_test_augmentation(size):
    """Add paddings to make image shape divisible by 32"""
    SIZE = size
    test_transform = [
        albu.Resize(SIZE,SIZE),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)