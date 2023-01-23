import collections
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from dataset import CustomDataset

# path = 'D:/Code/DataSets/CK+/CK+48'
def load_data(path):
    ckplus = {}
    for dir, subdir, files in os.walk(path):

        if files:
            sections = os.path.split(dir)
            emotion = sections[-1]

            # create child dictionary that groups images of the same subject id together
            subjects = collections.defaultdict(list)
            for file in files:
                # the subject id is present at the beginning of the file name
                subject = file.split("_")[0]
                subjects[subject].append(os.path.join(dir, file))

            ckplus[emotion] = subjects

    return ckplus


def prepare_data(data):
    # count number of images
    n_images = sum(len(paths) for paths in data.values())
    emotion_mapping = {emotion: i for i, emotion in enumerate(data.keys())}
    # {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'sadness': 5, 'surprise': 6}

    # 遍历字典，将key val 值返回
    # cla_dict = dict((val, key) for key, val in emotion_mapping.items())
    # cla_dict
    # {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'}

    # allocate required arrays
    image_array = np.zeros(shape=(n_images, 48, 48))
    image_label = np.zeros(n_images)

    i = 0
    for emotion, img_paths in data.items():
        for path in img_paths:
            image_array[i] = np.array(Image.open(path))  # load image from path into numpy array
            image_label[i] = emotion_mapping[emotion]  # convert emotion to its emotion id
            i += 1

    return image_array, image_label


def split_data(data):
    train = collections.defaultdict(list)
    test = collections.defaultdict(list)
    val = collections.defaultdict(list)

    for emotion, subjects in data.items():

        # shuffle each emotion's subjects and split them using a 0.8, 0.1, 0.1 split
        # subjects_train, subjects_test = train_test_split(list(subjects.keys()), test_size=0.2, random_state=1, shuffle=True)
        # subjects_train, subjects_val = train_test_split(subjects_test, test_size=0.5, random_state=1, shuffle=True)  # 0.2 * 0.8 = 0.1
        subjects_train, subjects_test = train_test_split(list(subjects.keys()), test_size=0.2, shuffle=True)
        subjects_test, subjects_val = train_test_split(subjects_test, test_size=0.5, shuffle=True)  # 0.2 * 0.8 = 0.1

        for subject in subjects_train:
            train[emotion].extend(subjects[subject])

        for subject in subjects_val:
            val[emotion].extend(subjects[subject])

        for subject in subjects_test:
            test[emotion].extend(subjects[subject])

    return train, val, test


def get_dataloaders(path, bs):
    # path = 'D:/Code/DataSets/CK+/CK+48'
    ckplus = load_data(path)
    train, val, test = split_data(ckplus)

    xtrain, ytrain = prepare_data(train)
    xval, yval = prepare_data(val)
    xtest, ytest = prepare_data(test)

    mu, st = 0, 1
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(mu,), std=(st,))

        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(
            lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
    ])

    test_transform = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(mu,), std=(st,))

        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(
            lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=12)
    valloader = DataLoader(val, batch_size=bs, shuffle=True, num_workers=12)
    testloader = DataLoader(test, batch_size=bs, shuffle=True, num_workers=12)

    return trainloader, valloader, testloader



















