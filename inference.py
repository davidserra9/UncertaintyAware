import os
import cv2
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf import OmegaConf
from torchvision.transforms.functional import to_pil_image
from src.logging import logger
from src.models import get_model
from src.ICM_dataset import get_validation_augmentations
from src.MC_wrapper import MCWrapper

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes
def main(cfg) -> None:

    if "skip_classes" in cfg.inference:
        classes_to_skip = []
        for cls in cfg.inference.skip_classes:
            try:
                classes_to_skip.append(cfg.base.classes.index(cls))
            except:
                logger.warn(f"Class {cls} not found in the list of classes at the base configurations.")
        cfg.inference.skip_classes = classes_to_skip

    # Find which device is used
    if torch.cuda.is_available() and cfg.base.device == "cuda":
        logger.info(f'Running inference in {torch.cuda.get_device_name(torch.cuda.current_device())}')
    else:
        logger.warn('CAREFUL!! Training the model with CPU')

    if "weights" not in cfg.model.encoder.params:
        raise ValueError("You must specify the path to the weights to load in the config file")
    else:
        model = get_model(cfg.model.encoder)
        model = model.eval()
        model = model.to(cfg.base.device)
        mc_model = MCWrapper(model,
                             num_classes=len(cfg.base.classes),
                             mc_samples=cfg.uncertainty.mc_samples,
                             dropout_rate=cfg.uncertainty.dropout_rate)

    if "class_activation_maps" in cfg.inference:
        import torchcam
        cam_model = getattr(torchcam.methods, cfg.inference.class_activation_maps)(model)
        os.makedirs(os.path.join(logger.output_path, "class_activation_maps"), exist_ok=True)

    # Obtain image paths
    img_paths = []
    for entry in os.scandir(cfg.inference.input_path):
        if entry.is_file() and entry.name.lower().endswith(tuple(IMG_FORMATS)):
            img_paths.append(entry.path)

    # Obtain video paths
    vid_paths = []
    for entry in os.scandir(cfg.inference.input_path):
        if entry.is_file() and entry.name.lower().endswith(tuple(VID_FORMATS)):
            vid_paths.append(entry.path)

    logger.info(f"Found {len(img_paths)} images and {len(vid_paths)} videos in {cfg.inference.input_path}")

    transforms = get_validation_augmentations()

    # Run inference on images
    df = pd.DataFrame(columns=["ID", "Path", "Prediction", "Uncertainty"])
    for img_path in tqdm(img_paths, desc="Inference on images", leave=False):
        input_tensor = transforms(image=cv2.imread(img_path)[:, :, ::-1])['image'].unsqueeze(0)
        input_tensor = input_tensor.to(cfg.base.device)

        prediction, uncertainty = mc_model(input_tensor)

        if "class_activation_maps" in cfg.inference:
            output = model(input_tensor)
            cam = cam_model(prediction, output)
            result = np.array(torchcam.utils.overlay_mask(Image.open(img_path), to_pil_image(cam[0].squeeze(0), mode='F'), alpha=0.8))
            cv2.imwrite(os.path.join(logger.output_path, "class_activation_maps", f"{os.path.basename(img_path).split('.')[0]}_cam.jpg"), result[:, :, ::-1])

        if int(prediction) not in cfg.inference.skip_classes:
            df.loc[len(df.index)] = [os.path.basename(img_path).split('.')[0], img_path, cfg.base.classes[int(prediction)], uncertainty]

    # save results
    df.to_excel(os.path.join(logger.output_path, "image_results.xlsx"), index=False)

    # Run inference on videos
    df = pd.DataFrame(columns=["ID", "Path", "Frame", "Timestamp", "Prediction", "Uncertainty"])
    for idx, vid_path in enumerate(vid_paths):
        vidcap = cv2.VideoCapture(vid_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        if "class_activation_maps" in cfg.inference:
            vidout = cv2.VideoWriter(os.path.join(logger.output_path, "class_activation_maps", f"{os.path.basename(vid_path).split('.')[0]}_cam.mp4"),
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps,
                                     (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        for frame in tqdm(range(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=f"Inference on video {idx+1}/{len(vid_paths)} ", leave=False):
            success, image = vidcap.read()
            if success:
                input_tensor = transforms(image=image)['image'].unsqueeze(0)
                input_tensor = input_tensor.to(cfg.base.device)

                prediction, uncertainty = mc_model(input_tensor)

                if prediction not in cfg.inference.skip_classes:
                    df.loc[len(df.index)] = [os.path.basename(vid_path).split('.')[0],
                                             vid_path,
                                             frame+1,
                                             pd.to_timedelta(frame/fps, unit='s'),
                                             cfg.base.classes[int(prediction)],
                                             uncertainty]

                    if "class_activation_maps" in cfg.inference:
                        output = model(input_tensor)
                        cam = cam_model(prediction, output)
                        result = np.array(torchcam.utils.overlay_mask(Image.fromarray(image), to_pil_image(cam[0].squeeze(0), mode='F'), alpha=0.8))
                        vidout.write(result[..., ::-1])

                else:
                    if "class_activation_maps" in cfg.inference:
                        vidout.write(image[..., ::-1])

            else:
                logger.error(f"Error reading frame {frame} from video {vid_path}")

        if "class_activation_maps" in cfg.inference:
            vidout.release()

    df['Timestamp'] = df['Timestamp'].apply(lambda x: str(x).split('.')[0])
    df.to_excel(os.path.join(logger.output_path, "video_results.xlsx"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on a set of images and videos with a pretrained model.")
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args(sys.argv[1:])
    config_name = args.config

    initialize(version_base=None, config_path="config", job_name="training")
    config = compose(config_name=config_name)
    config = OmegaConf.create(config)
    main(config)