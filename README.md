# MAIA - AIA-ML-DL - Calcification Detection

## Team Members

- ### Cortina Uribe Alejandro

- ### Seia Joaquin Oscar

- ### Zalevskyi Vladyslav

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

## To get and turn to a useful format the INBreast database

```bash
cd data &&
gdown https://drive.google.com/uc?id=1ebw9N2vZY19TuELBZb39eAJhPjY1eFZX &&
unzip 'INbreast Release 1.0.zip' &&
rm -rf 'INbreast Release 1.0.zip' &&
cd ../ &&
python database/parsing_metadata.py --ib-path data/INbreast\ Release\ 1.0/ --rp --cb --pect-musc-mask
```

## To get the checkpoints of pretrained models

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
gdown https://drive.google.com/uc?id=1v-nDrdt2ejno7QVZvgRbqIVM27ymx7ft &&
unzip ml_cascade_checkpoints.zip &&
rm -rf ml_cascade_checkpoints.zip &&
cd ../../
```

### Example image

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

#### Suggestion for contributers

- numpy docstring format
- flake8 lintern
- useful VSCode extensions:
  - autoDocstring
  - Python Docstring Generator
  - GitLens
