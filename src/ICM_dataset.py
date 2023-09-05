import os
import cv2
import torch
from os.path import join
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import random

class ICMDataset(Dataset):
    """ Custom dataset class for loading images and labels from a list of directories divided in splits """

    def __init__(self, path, train, species, oversample=False):
        self.images = []
        self.labels = []
        self.train = train
        self.transforms = get_training_augmentations() if train else get_validation_augmentations()

        random.seed(42)

        if not train:   # Validation/Test
            for folder_path in glob(join(path, "*")):
                if os.path.isdir(folder_path):
                    for annot_path in glob(join(folder_path, "*0.jpg")):
                        self.images.append([annot_path[:-5] + str(i) + annot_path[-4:] for i in range(5)])
                        self.labels.append(species.index(folder_path.split("/")[-1]))

        elif not oversample:    # Training without oversampling
            for folder_path in glob(join(path, "*")):
                if os.path.isdir(folder_path):
                    for annot_path in glob(join(folder_path, "*.jpg")):
                        self.images.append(annot_path)
                        self.labels.append(species.index(folder_path.split("/")[-1]))

        else:   # Training with oversampling
            max_annotations = 0  # Find the max number of images of a class
            for folder_path in glob(join(path, "*")):
                if os.path.isdir(folder_path):
                    max_annotations = max(max_annotations, len(glob(join(folder_path, "*0.jpg"))[:10]))

            for folder_path in glob(join(path, "*")):
                if os.path.isdir(folder_path):
                    species_annot = len(glob(join(folder_path, "*0.jpg")))
                    annot_counter = 0
                    for _ in range(-(max_annotations // -species_annot)):
                        for annot_path in glob(join(folder_path, "*0.jpg")):
                            self.images += [annot_path[:-5] + str(i) + annot_path[-4:] for i in range(5)]
                            self.labels += [species.index(folder_path.split("/")[-1])] * 5
                            annot_counter += 1
                            if annot_counter == max_annotations:
                                break

    def __len__(self) -> int:
        """ Length of the dataset. """
        return len(self.images)

    def __getitem__(self, index):
        """ Get the item at the given index. """
        if self.train:
            # Read image and transform it
            img = cv2.imread(self.images[index])[:, :, ::-1]
            img = self.transforms(image=img)['image']

            # Obtain the label and encode them
            label = self.labels[index]

            return img, torch.tensor(label)

        else:
            # Read images and transform them
            image_paths = self.images[index]
            images = [self.transforms(image=cv2.imread(path)[:, :, ::-1])['image'] for path in image_paths]
            images = torch.stack(images)

            # Obtain the label and encode them
            label = self.labels[index]

            return images, torch.tensor(label)

def get_training_augmentations():
    """ Function defining and returning the training augmentations.

    Returns
    -------
    train_transform : albumentations.Compose
        training augmentations
    """
    train_transform = [
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=5, p=0.2),
            A.Blur(blur_limit=5, p=0.2),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0.1, rotate_limit=10, p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(),
        ], p=0.2),
        A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                    std=[0.1263, 0.1265, 0.1169]),
        A.Resize(224, 224),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


def get_validation_augmentations():
    """ Function defining and returning the validation/test augmentations.

        Returns
        -------
        val_transforms : albumentations.Compose
            training augmentations
    """
    val_transforms = [
        A.Normalize(mean=[0.4493, 0.5078, 0.4237],
                    std=[0.1263, 0.1265, 0.1169]),
        A.Resize(224, 224),
        ToTensorV2(),
    ]
    return A.Compose(val_transforms)
