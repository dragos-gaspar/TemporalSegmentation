import logging
import os
import ssl

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import ModelConfig as Config
from model.SiameseNet import SiameseNet
from model.resnet import resnet18


logger = logging.getLogger('model')
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel('DEBUG')


ssl._create_default_https_context = ssl._create_unverified_context


def draw_pair(frame1, frame2):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(frame1)
    axs[1].imshow(frame2)
    plt.show()


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def draw_pair(i, label, frame1, frame2):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(frame1)
    axs[1].imshow(frame2)
    plt.savefig('images/'+str(label)+'_'+str(i)+'.png')
    # plt.show()


def get_feature_extractor():
    logger.info('Downloading resnet18 weights...')
    resnet_backbone = resnet18(pretrained=True)
    model = SiameseNet(backbone=resnet_backbone)
    return model


def train():
    dataset = []
    labels = []
    transitions_path = os.path.join(Config.DATA_PATH, 'transitions')
    for data in sorted(os.listdir(transitions_path)):
        dataset.append(np.transpose(np.load(os.path.join(transitions_path, data)), (0, 3, 1, 2)))
        # dataset.append(np.load(os.path.join(transitions_path, data)))
        labels.append(1)
    non_transition_path = os.path.join(Config.DATA_PATH, 'non-transitions')
    for data in sorted(os.listdir(non_transition_path)):
        dataset.append(np.transpose(np.load(os.path.join(non_transition_path, data)), (0, 3, 1, 2)))
        # dataset.append(np.load(os.path.join(non_transition_path, data)))

        labels.append(0)

    model = get_feature_extractor()
    distances = []
    for i, data in enumerate(dataset):
        logger.info(f'Preprocessing image {i}...')

        data = torch.tensor(data)
        t1 = preprocess(data[0])
        t2 = preprocess(data[1])

        out = model(t1.unsqueeze(0), t2.unsqueeze(0))
        distances.append(out.item())

    [X_train, X_test, y_train, y_test] = train_test_split(np.array(distances).reshape(-1, 1), labels)

    clf = svm.SVC()
    logger.info('Training...')
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    logger.info(f'Validation accuracy is {accuracy_score(y_test, pred)}')


def predict():
    ...
