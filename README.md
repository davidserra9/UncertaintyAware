# Uncertainty Aware Deep Learning Classification of Marine Species from ROV Vehicle Videos

[David Serrano](https://scholar.google.es/citations?user=CWuYYNUAAAAJ&hl=en&oi=sra), [David Masip](https://scholar.google.es/citations?user=eHOqwS8AAAAJ&hl=en&oi=ao), José A. Garcia del Arco, Montserrat Demestre, Sara Soto, Andrea Cabrito, Laia Illa-López

[AIWell Research Group, Universitat Oberta de Catalunya](https://aiwell.uoc.edu/)

[Insitut de Ciències del Mar, Spanish National Research Council (CSIC)](https://icm.csic.es/en)

This repositoy contains all the code and software of the paper "Uncertainty Aware Deep Learning Classification of Marine Species from ROV Vehicle Videos". The software is designed to classify Marine Species using MonteCarlo Dropout to generate predictions along with associated estimates of uncertainty.

The repository contains the code to replicate the experiments conducted in the paper. The experiments focus on uncertainty estimations and utilize a novel dataset of ROV vehicle images from the Mediterranean Sea, called ICM-20. The results demonstrate that incorporating uncertainty estimates can improve the utilization of human annotators' efforts when correcting misclassified samples.

Additionally, the repository includes the code to generate the Correct versus Incorrent Histogram (CIH) and the Accuracy versus Corrected Curve (ACC) proposed in the paper.

## Requirements
Python >= 3.6, Torchvision >= 0.13.0 (to use ConvNeXt), Pytorch, Numpy,  

## Installation
To install the required dependencies, create a conda environment. 
Create a conda environment named `uncertainty` and install the dependencies.

## Usage
### Configuration files
The pipeline uses `.yaml` config files to set the parameters for various aspects of the code. These config files are stored in the `config` folder and are divided into five categories: `base`, `model`, `training`, `uncertainty` and `inference`. Each category corresponds to specific parameters related to the respective process.

The parent config file groups the individual files for each category, as shown below:

```yaml
defaults:
  - base: base_example
  - model: efficientnet_b0_example
  - training: training_example
  - uncertainty: uncertainty_example
  - inference: inference_example
```

You can modify teh values in the individual config files or create new ones based on the provided examples.

### Log files
During the execution of the code, a log folder will be created in the repository. Each run will generate a subfolder with a timestamp, containing all the files generated during that run. The log folder will always contain a `log.log` file, which provides information about the state fo the run. Additionally, the log files specific to each type of run are organized as follows:
```
📂logs/
├── 📂YYYY-MM-DD_HH-MM-SS/ (train.py log)
│   ├── 📜log.log
│   ├── 📜config.yaml
│   ├── 💾epoch_0_validacc_0.5_validf1_0.5.pt
│   ├── 💾 ...
│   ├── 💾epoch_X_validacc_X_validf1_X.pt
│   ├── 🖼️valid_confusion_matrix.jpg
│   ├── 🖼️ACC.jpg
│   ├── 🖼CIH.jpg
│
├── 📂YYYY-MM-DD_HH-MM-SS/ (eval.py log)
│   ├── 📜log.log
│   ├── 📜config.yaml
│   ├── 🖼️test_confusion_matrix.jpg
│   ├── 🖼️ACC.jpg
│   ├── 🖼CIH.jpg
│
├── 📂YYYY-MM-DD_HH-MM-SS/ (inference.py log)
    ├── 📜log.log
    ├── 📜config.yaml
    ├── 🧮video_results.xlsx
    ├── 🧮image_results.xlsx
    ├── 📂class_activation_maps.jpg
        ├── 🖼image1_cam.jpg
        ├── 🖼video1_cam.mp4
```

### Dataset
The ICM-Benchmark-20 dataset is available at Kaggle:
[https://www.kaggle.com/datasets/tsunamiserra/icm-benchmark-20](https://www.kaggle.com/datasets/tsunamiserra/icm-benchmark-20)

The ICM-20 is available under request. Please contact [jagarcia@icm.csic.es](jagarcia@icm.csic.es) for obtaining the download instructions.
The database was created from 352GB of videos, so it is quite difficult to manage. Each of the 17 videos have an excel file with all the annotation with their corresponding timestamp and video frame.

### Training a model
To train a model, run the following command. This will train a model from scratch if the parameter `model.encoder.params.weights` is not specified. If the parameter is specified, the model will load the weights from the specified path and train from those pretrained weights. The model weights will be saved in the log folder.

```bash
python train.py --config config/config.yaml
```

### Evaluating a model
To evaluate a model, run the following command. This will evaluate the model specified on `model.encoder.params.weights` on the test set and save the results in the log folder.

```bash
python eval.py --config config/config.yaml
```

### Inference
To run inference on a model, run the following command. This will run inference using the model specified on `model.encoder.params.weights` on all the images and videos in the `inference.input_path`. If the parameter `inference.class_activation_maps` is set to any method, the input images or videos will be stored in the log file with the class activation maps heatmaps overlayed. The results will be stored in the log folder. This script also saves an excel file with the predictions and uncertainty estimations for each image and video.

```bash
python inference.py --config config/config.yaml
```

Feel free to modify the configuration files to adapt the parameters based on your specific needs.