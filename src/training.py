import os
import random
import cv2
import torch
import wandb
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from glob import glob
from datetime import datetime, timedelta
from src.models import save_model
from src.metrics import compute_metrics, predictive_entropy, uncertainty_box_plot, uncertainty_curve
from src.logging import logger
from src.MC_wrapper import MCWrapper
from src.CAM_wrapper import AM_initializer, append_maps
from src.ICM_dataset import get_validation_augmentations

def get_optimizer(model, cfg):
    if cfg.name.lower() == "sgd":
        # lr, wegith_decay, momentum
        optimizer = torch.optim.SGD(model.parameters(), **cfg.params)
        logger.info(f"Using SGD optimizer w/ {cfg.params}")
    elif cfg.name.lower() == "adam":
        # lr, weight_decay
        optimizer = torch.optim.Adam(model.parameters(), **cfg.params)
        logger.info(f"Using ADAM optimizer w/ {cfg.params}")

    else:
        logger.error("Optimizer not implemented")
        raise ValueError("Optimizer not implemented")

    return optimizer

def get_scheduler(optimizer, cfg):
    if cfg.name.lower() == "exponentiallr":
        # gamma
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **cfg.params)
        logger.info(f"Using ExponentialLR scheduler w/ {cfg.params}")

    elif cfg.name.lower() == "cosineannealinglr":
        # T_max, eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cfg.params)
        logger.info(f"Using CosineAnnealingLR scheduler w/ {cfg.params}")

    elif cfg.name.lower() == "lambdalr":
        # lr_lambda
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: cfg.params.lr_lambda ** epoch)
        logger.info(f"Using LambdaLR scheduler w/ {cfg.params}")

    elif cfg.name.lower() == "none":
        scheduler = None
        logger.info("No scheduler used")
    else:
        logger.error(f"{cfg.name} scheduler not implemented")
        raise ValueError("Scheduler not implemented")
    return scheduler

def train_epoch(model, train_loader, criterion, optimizer, scheduler, log_step, epoch, wb_log, device):
    model.train()
    running_loss, total_samples, correct_samples = 0.0, 0, 0
    with tqdm(train_loader, unit="batch", leave=False, desc=f"TRAIN {epoch}") as pbar:
        for idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()               # Initialize the gradients
            outputs = model(inputs)             # Forward pass
            _, preds = torch.max(outputs, 1)    # Predictions

            loss = criterion(outputs, labels)   # Compute the loss
            loss.backward()                     # Backward pass
            optimizer.step()                    # Update the weights
            if scheduler is not None:
                scheduler.step()                    # Update the learning rate

            total_samples += labels.size(0)
            correct_samples += (preds == labels).sum().item()
            running_loss += loss.item()

            if idx % log_step == 0 and idx != 0:
                pbar.set_postfix({"lr": optimizer.param_groups[0]['lr'],
                                  "loss": running_loss / (idx+1),
                                  "acc": correct_samples / total_samples})

            if wb_log and idx % log_step == 0 and idx != 0:
                wandb.log({"train/loss": running_loss / (idx+1),
                           "train/acc": correct_samples / total_samples,
                           "train/lr": optimizer.param_groups[0]['lr']})

def valid_epoch(model, valid_loader, criterion, log_step, epoch, wb_log, cls_names, device):
    model.eval()
    running_loss = 0.0
    predictions, targets = np.empty(0), np.empty(0)
    with torch.no_grad():
        with tqdm(valid_loader, unit="batch", leave=False, desc=f"VALID {epoch}") as pbar:
            for idx, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.squeeze())
                outputs = torch.mean(outputs, dim=0, keepdim=True)
                _, preds = torch.max(outputs, 1)

                predictions = np.append(predictions, preds.cpu().numpy(), axis=0)
                targets = np.append(targets, labels.cpu().numpy(), axis=0)

                loss = criterion(outputs, labels)
                running_loss += loss.item()

                if idx % log_step == 0 and idx != 0:
                    pbar.set_postfix({"loss": running_loss / (idx+1)})

    f1, acc, cm = compute_metrics(targets, predictions, cls_names)
    if wb_log:
        wandb.log({"valid/loss": running_loss / (idx+1),
                   "valid/acc": acc,
                   "valid/f1": f1,
                   "valid/cm": wandb.Image(cm)})

    return acc, f1, running_loss / (idx+1), cm

def fit(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, wb_log, log_step, cls_names, output_path, device):
    max_acc, max_f1, min_loss = 0.0, 0.0, 1000000.0
    logger.info(f"===== Starting training =====")
    start = time.time()
    for epoch in range(epochs):
        train_epoch(model, train_loader, criterion, optimizer, scheduler, log_step=log_step, epoch=epoch, wb_log=wb_log, device=device)
        acc, f1, loss, cm = valid_epoch(model, valid_loader, criterion, log_step=log_step, epoch=epoch, wb_log=wb_log, cls_names=cls_names, device=device)

        msg = f" Epoch {epoch:02} | acc: {acc:.4f} - f1: {f1:.4f} - loss: {loss:.4f}"
        if f1 > max_f1:
            max_f1 = f1
            max_acc = acc
            min_loss = loss
            model_path = os.path.join(output_path, f"epoch_{epoch:0{len(str(epochs))}}_validacc_{acc:.4}_validf1_{f1:.4}.pt")
            save_model(model, optimizer, epoch, acc, f1, model_path)
            msg += " | Model saved @ {}".format(model_path)

            cm.savefig(os.path.join(output_path, f"valid_confusion_matrix.jpg"))

        logger.info(msg)
    logger.info(f"=== Training finished in {timedelta(seconds=round(time.time() - start))} w/ ACC: {max_acc:.4f}, F1: {max_f1:.4f}, LOSS: {min_loss:.4f} ===")

    if wb_log:
        wandb.summary["best_acc"] = max_acc
        wandb.summary["best_f1"] = max_f1
        wandb.summary["best_loss"] = min_loss


def eval_uncertainty_model(model, eval_loader, mc_samples, dropout_rate, num_classes, wb_log, output_path, device):
        mc_wrapper = MCWrapper(model, num_classes=num_classes, mc_samples=mc_samples, dropout_rate=dropout_rate)

        pred_y, true_y, pred_unc = np.array([], dtype=np.uint8), np.array([], dtype=np.uint8), np.array([])

        # Iterate over the loader and stack all the batches predictions
        for (batch, target) in tqdm(eval_loader, desc="Uncertainty with MC Dropout", leave=False):
            batch, target = batch.to(device), target.to(device)
            for b, t in zip(batch, target.cpu().numpy()):
                prediction, uncertainty = mc_wrapper(b)
                pred_y = np.append(pred_y, prediction)
                pred_unc = np.append(pred_unc, uncertainty)
                true_y = np.append(true_y, t)

        box_plot, intersection = uncertainty_box_plot(y_true=true_y, y_pred=pred_y, entropy=pred_unc)
        curve, au, nau = uncertainty_curve(y_true=true_y, y_pred=pred_y, ent=pred_unc)

        logger.info(f"MC Dropout | CII: {intersection:.4f} - AUC: {au:.4f} - NAUC: {nau:.4f}")

        if wb_log:
            wandb.log({"eval/box_plot": wandb.Image(box_plot),
                       "eval/curve": wandb.Image(curve)})
            wandb.summary["eval/au"] = au
            wandb.summary["eval/nau"] = nau

        box_plot.savefig(os.path.join(output_path, "CIH.jpg"))
        curve.savefig(os.path.join(output_path, "ACC.jpg"))

def inference_images(model, num_images, technique, mc_samples, dropout_rate, images_path, cls_names, device):
    model.eval()
    mc_wrapper = MCWrapper(model, num_classes=len(cls_names), mc_samples=mc_samples, dropout_rate=dropout_rate)
    cam_wrapper = AM_initializer(model, technique)

    transforms = get_validation_augmentations()

    # Obtain "num_images" images from each class
    cls_paths = glob(os.path.join(images_path, "*"))
    images = {}
    for cls_path in cls_paths:
        cls_name = cls_path.split("/")[-1]
        images[cls_name] = []
        cls_images = glob(os.path.join(cls_path, "*2.jpg"))
        random.shuffle(cls_images)
        images[cls_name] = cls_images[:num_images]

    for cls_name, cls_images in images.items():
        for idx, img_path in enumerate(tqdm(cls_images, desc=f"Inference {cls_name}", leave=False)):
            # Load images
            img = cv2.imread(img_path)
            tensor = transforms(image=img)["image"]
            tensor = tensor.unsqueeze(0).to(device)

            outputs = mc_wrapper(tensor)
            mean = np.mean(outputs, axis=1)

            pred_y = mean.max(axis=1).argmax(axis=-1)
            pred_entropy = predictive_entropy(mean)

            heatmap = cam_wrapper(tensor, idx=pred_y)[0]
            map = append_maps(img, heatmap)

            results = np.vstack((img/255, map))

            plt.imshow(results)
            plt.title(f"Class: {cls_name} - Pred: {cls_names[int(pred_y)]} - Entropy: {pred_entropy:.4f}", fontsize=5)
            plt.axis("off")
            wandb.log({f"inference/{cls_name}": wandb.Image(plt.gcf())})
            plt.close()
