# -*- coding: utf-8 -*-
"""
This script contains the model wrapper which ease the uncertainty estimation from Monte-Carlo dropout method.
It uses a lot of matrix computations to speed up the forward passes.
The implemented architectures are:
    - VGG
    - ResNet
    - EfficientNet
    - EfficientNetV2
    - ConvNeXt
"""
import sys
import torch
import numpy as np
import torch.nn as nn


class MCWrapper(object):
    """Monte-Carlo Dropout model wrapper
    Implemented Architectures: (if the architecture does not have a dropout layer, then it is added at inference time)
        - VGG
        - ResNet
        - EfficientNet
        - EfficientNetV2
        - ConvNeXt
    """

    def __init__(self, model, num_classes, mc_samples=25, dropout_rate=0.5, device="cuda"):

        self.model = model.eval()
        self.dropout_rate = dropout_rate
        self.samples = mc_samples
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        self.device = device

        # VGG: dropout layer in train mode
        if 'vgg' in model.name:
            self.model = self.train_dropout(self.model)

        # ResNet: append dropout layer in the classifier in train mode
        elif 'resnet' in model.name:
            self.model = self.append_dropout_resnet(self.model)
            self.model = self.train_dropout(self.model)

        # EfficientNet: dropout layer in train mode
        elif 'efficientnet_b' in model.name:
            self.model = self.train_dropout(self.model)

        # EfficientNetV2: dropout layer in train mode
        elif 'effificnet_v2_b' in model.name:
            self.model = self.train_dropout(self.model)

        # ConvNeXt: Add dropout layer in the classifier in train mode
        elif 'convnext' in model.name:
            self.model.classifier = self.change_dropout_rate(self.model.classifier)
            self.model = self.train_dropout(self.model)

    def append_dropout_resnet(self, model):
        """ Append dropout layer to the resnet classifier

            Parameters
            ----------
            model : torch.nn.Module
                ResNet model (with classifier module)
        """

        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.append_dropout_resnet(module)
            if name == 'layer4':
                new = nn.Sequential(module, nn.Dropout2d(p=self.dropout_rate, inplace=True))
                setattr(model, name, new)
        return model

    def train_dropout(self, model):
        """ Function to put dropout layers in training mode

            Parameters
            ----------
            model : torch.nn.Module
        """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        return model

    def change_dropout_rate(self, model):
        """ Change the dropout rate of the model

            Parameters
            ----------
            model : torch.nn.Module
        """
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.change_dropout_rate(module)
            if name == 'drop':
                new = nn.Sequential(module, nn.Dropout2d(p=self.dropout_rate, inplace=True))
                setattr(model, name, new)
        return model

    def predictive_entropy(self, mean):
        """ Predictive entropy of the dropout mean

            Parameters
            ----------
            dropout_mean : np.array
                Array with shape (B, L)
                B: batch size or number of images in the dataloader
                L: number of classes
        """
        return -np.sum(mean * np.log(mean + sys.float_info.min))

    def forward(self, x):
        """ N forward passes of the model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, H, W)
            B: batch size
            C: number of channels
            H: height
            W: width

        Return
        ------
        np.array
            Array with shape (B, S, L)
            B: batch size or number of images in the dataloader
            S: number of monte-carlo samples
            L: number of classes
        """
        dropout_predictions = np.empty((self.samples, x.shape[0], self.num_classes))
        for i in range(self.samples):
            with torch.no_grad():
                outputs = self.model(x)
                #outputs = torch.stack([self.model(x[i, :, :, :, :].to(self.device)) for i in range(x.shape[0])])
                outputs = self.softmax(outputs)

            dropout_predictions[i] = outputs.cpu().numpy()

        dropout_mean = np.mean(dropout_predictions, axis=0)

        if len(dropout_mean.shape) > 1: # If more than one image per annotation
            dropout_mean = np.mean(dropout_mean, axis=0)

        return dropout_mean.argmax(axis=0).item(), self.predictive_entropy(dropout_mean)

    def __call__(self, x):
        return self.forward(x)
