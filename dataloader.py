import os
import PIL

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


def only_mixup(num_classes=100):
    return T.MixUp(num_classes=num_classes)


def only_cutmix(num_classes=100):
    return T.CutMix(num_classes=num_classes)


def mixup_cutmix(num_classes=100):
    mixup = T.MixUp(num_classes=num_classes)
    cutmix = T.CutMix(num_classes=num_classes)
    return T.RandomChoice([mixup, cutmix])


def train_transform():
    ColorJitter = T.ColorJitter(brightness=0.4, contrast=0.4,
                                saturation=0.4, hue=0.1)

    train_transform_list = T.Compose([
        T.RandomResizedCrop(320),
        T.RandomHorizontalFlip(),
        T.RandomApply([ColorJitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train_transform_list


def valid_transform():
    valid_transform_list = T.Compose([
        T.Resize(512),
        T.CenterCrop(320),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return valid_transform_list


def load_dataset(folder='data', usage='train'):
    assert usage in ['train', 'val', 'test']
    path = os.path.join(folder, usage)

    if usage == 'test':
        return [os.path.join(path, file) for file in os.listdir(path)]

    imgs = []
    labels = []

    for label in os.listdir(path):
        cur_path = os.path.join(path, label)

        for img_file in os.listdir(cur_path):
            imgs.append(os.path.join(cur_path, img_file))
            labels.append(int(label))

    return imgs, labels


class Image_Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
