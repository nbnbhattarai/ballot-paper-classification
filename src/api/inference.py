'''
Module containing model inference views
'''

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

N_CLASSES = 48


def classify_ballot_paper(image):
    '''
    Classify ballot paper
    '''
    # model_saved = open('../models/model.tk')

    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, N_CLASSES, bias=True),
    )

    # model.load(model_saved)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_t = data_transform(image).unsqueeze(0)

    model.eval()
    pred = model(image_t)

    pred_class = torch.argmax(pred, dim=1)

    print(pred_class)

    return pred_class
