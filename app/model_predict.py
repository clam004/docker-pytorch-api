#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from config import CONFIG

class Model(nn.Module):
    """
    An example pytorch model for classifying iris flower
    """

    def __init__(self, input_dim=4, output_dim=3):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

def preprocess(package: dict, input: list) -> list:
    """
    Preprocess data before running with model, for example scaling and doing one hot encoding
    :param package: dict from fastapi state including model and preocessing objects
    :param package: list of input to be proprocessed
    :return: list of proprocessed input
    """

    # scale the data based with scaler fit during training
    scaler = package['scaler']
    input = scaler.transform(input)

    return input


def predict(package: dict, input: list) -> np.ndarray:
    """
    Run model and get result
    :param package: dict from fastapi state including model and preocessing objects
    :param package: list of input values
    :return: numpy array of model output
    """

    # process data
    X = preprocess(package, input)

    # run model
    model = package['model']
    model = model.to(CONFIG['DEVICE'])

    with torch.no_grad():
        # convert input from list to Tensor
        X = torch.Tensor(X)

        # move tensor to device
        X = X.to(CONFIG['DEVICE'])

        # run model
        y_pred = model(X)

    # convert result to a numpy array on CPU
    y_pred = y_pred.cpu().numpy()

    return y_pred