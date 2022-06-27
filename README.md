# Advance Image Analysis & Machine and Deep Learning Final Project

# Calcification Detection in Mammography Images

## MAIA Master 2020

---------------------------------------
## Team Members


- ### Cortina Uribe Alejandro

- ### Seia Joaquin Oscar

- ### Zalevskyi Vladyslav


## Instructions

This repository contains all the code and analysis notebooks for our final project on microcalcifications (MC) detection on mammography images that we develpped for our AIA and ML-DL courses. The project included the develpment of three different pipelines for MC detection, one based on AIA+ML and two deep learning based. The full report can be checked [here](https://github.com/joaco18/calc-det/final_report.pdf).

The main final results analysis notebooks are:

- On **validaton set**:
  
  - [Adavanced Image Analysis and Machine Learning](https://github.com/joaco18/calc-det/blob/dev/notebooks/detection_by_aia_plus_ml_analysis.ipynb)
  
  - [Deep Learning - Detection by classification](https://github.com/joaco18/calc-det/blob/dev/notebooks/colab/detection_by_classification_analysis.ipynb)
  
  - [Deep Learning - Detection with FasterRCNN](https://github.com/joaco18/calc-det/blob/dev/notebooks/colab/detection_by_fasterrcnn_analysis.ipynb)

  - [Three pipelines - best models ](https://github.com/joaco18/calc-det/blob/dev/notebooks/colab/final_comparison_between_all_methods_val_set.ipynb)

- On **test set**:
  - [Three pipelines - best models](https://github.com/joaco18/calc-det/blob/dev/notebooks/colab/final_comparison_between_all_methods_test_set.ipynb)

Even if the repository is self-contained and it be fully reproduced following the undegoing instructions, the reader might be interested first in checking and runing examples of the final pipelines we generated. To do so, you should do:

- Environmental set up
- Download the checkpoints for the models
- Download the example image (or provide one of your own)
- Run the examples

The mentionned steps are here provided in bash command line format:

## Environment set up

Start by creating a new conda environment

```bash
conda update -n base -c defaults conda &&
conda create -n calc_det anaconda &&
conda activate calc_det &&
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Download checkpoints of pretrained models

### Deep learning detection by classification

```bash
mkdir deep_learning/classification_models/checkpoints &&
cd deep_learning/classification_models/checkpoints &&
gdown https://drive.google.com/uc?id=16BbvvZcS2Qx421v9QKpKH4JrVKF1Efcf &&
unzip classification_checkpoints.zip &&
rm -rf classification_checkpoints.zip &&
cd ../../../
```

### Deep learning detection with FasterRCNN

```bash
mkdir deep_learning/detection_models/checkpoints &&
cd deep_learning/detection_models/checkpoints &&
gdown https://drive.google.com/uc?id=1R8fxd_CdyG5ec1grobRUut8UqCKbVFdp &&
unzip detection_checkpoints.zip &&
rm -rf detection_checkpoints.zip &&
cd ../../../
```

### Machine Learning

```bash
mkdir machine_learning/checkpoints &&
cd machine_learning/checkpoints &&
gdown https://drive.google.com/uc?id=1TOJ3nsXnxfvMxygeXGQHxKH5-qOXeTBL &&
unzip ml_cascade_checkpoints.zip &&
rm -rf ml_cascade_checkpoints.zip &&
cd ../../
```

## Download Example image

```bash
mkdir example_img &&
cd example_img &&
gdown https://drive.google.com/uc?id=1VYPWmU2QuEZ3Ys9LhAsDZp19dZmaaT4r &&
cd ../
```

## Runing a full case

### AIA-ML

```bash
python petunias_mc_detector.py --dcm-filepath <ABOSULTE_PATH_TO_REPO>/example_img/24065734_5291e1aee2bbf5df_MG_L_CC_ANON.dcm --detector-type 'aia_ml' --ouput-path /<ABOSULTE_PATH_TO_REPO>/example_img/ --store-csv --v
```

### Deep learning classification based detection

```bash
python petunias_mc_detector.py --dcm-filepath <ABOSULTE_PATH_TO_REPO>/example_img/24065734_5291e1aee2bbf5df_MG_L_CC_ANON.dcm --detector-type 'classification_dl' --ouput-path /<ABOSULTE_PATH_TO_REPO>/example_img/ --store-csv --v --batch-size 224
```

in colab don't pass '--batch-size'

### Deep learning detection based detection

```bash
python petunias_mc_detector.py --dcm-filepath <ABOSULTE_PATH_TO_REPO>/example_img/24065734_5291e1aee2bbf5df_MG_L_CC_ANON.dcm --detector-type 'detection_dl' --ouput-path /<ABOSULTE_PATH_TO_REPO>/example_img/ --store-csv --v --batch-size 1
```

in colab don't pass '--batch-size'

## Further instructions

If the reader wants to run the full code, then downloading and preparation of the INBreast should be done as following:

### Download and prrepare INBreast database

```bash
cd data &&
gdown https://drive.google.com/uc?id=1ebw9N2vZY19TuELBZb39eAJhPjY1eFZX &&
unzip 'INbreast Release 1.0.zip' &&
rm -rf 'INbreast Release 1.0.zip' &&
cd ../ &&
python database/parsing_metadata.py --ib-path data/INbreast\ Release\ 1.0/ --rp --cb --pect-musc-mask
```

<!-- #### Suggestion for contributers

- numpy docstring format
- flake8 lintern
- useful VSCode extensions:
  - autoDocstring
  - Python Docstring Generator
  - GitLens -->
