import os.path

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from packaging import version
from src.logging import logger

def get_model(cfg):
    if version.parse(torchvision.__version__) < version.parse("0.13.0"):
        logger.error("The torchvision version must be >= 0.13.0")
        raise ValueError("The torchvision version must be >= 0.13.0")

    if "efficientnet_b" in cfg.name:
        model = getattr(models, cfg.name)(weights=cfg.params.pretrained)

        model_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(model_ftrs, cfg.params.num_classes)
        model.name = cfg.name

    elif "efficientnet_v2_" in cfg.name:
        model = getattr(models, cfg.name)(weights=cfg.params.pretrained)

        model_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(model_ftrs, cfg.params.num_classes)
        model.name = cfg.name

    elif "convnext" in cfg.name:
        model = getattr(models, cfg.name)(weights=cfg.params.pretrained)

        model_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(model_ftrs, cfg.params.num_classes)
        model.name = cfg.name

    else:
        logger.error("Model not implemented")
        raise Exception("Model not implemented")

    st = f"Model {cfg.name} loaded"

    if "weights" in cfg.params:
        if not os.path.isfile(cfg.params.weights):
            logger.error(f"File {cfg.params.weights} does not exist. Not loading weights.")

        load_model(model, cfg.params.weights)
        st += f" with weights {cfg.params.weights}."

    logger.info(st)
    return model

def load_model(model, path):
    return model.load_state_dict(torch.load(path)["state_dict"])

def save_model(model, optimizer, num_epoch, acc, f1, path):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": num_epoch,
        "acc": acc,
        "f1": f1
    }

    torch.save(checkpoint, path)

