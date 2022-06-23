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

## To get the database

1.Download the raw database from the original [Drive](https://drive.google.com/file/d/19n-p9p9C0eCQA1ybm6wkMo-bbeccT_62/view) from the authors.

2.Extract the zipped images in a "data" directory inside this same directory, the directory structure should be the following:

```
calc-det
    └── data
        └── INbreast Release 1.0
                ├── AllDICOMs
                |    ├── files.dcm
                |    └── ...
                ├── AllROI
                |    ├── files.roi
                |    └── ...
                ├── AllXML
                |    ├── files.xml
                |    └── ...
                ├── MedicalReports
                |    ├── files.txt
                |    └── ...
                ├── PectoralMuscle
                |    ├── Pectoral Muscle ROI
                |    |   └── files.roi
                |    └── Pectoral Muscle XML
                |        └── files.xml
                ├── INbreast.csv
                ├── inreast.pdf
                ├── INreast.xls
                └── README.txt
```

3.Run the following command:

```bash
python database/parsing_metadata.py --ib-path data/INbreast\ Release\ 1.0/ --rp --v --cb --pect-musc-mask
```

## To get the checkpoints of pretrained models

### Deep learning detection by classification

```bash
cd deep_learning/classification_models &&
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16BbvvZcS2Qx421v9QKpKH4JrVKF1Efcf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16BbvvZcS2Qx421v9QKpKH4JrVKF1Efcf" -O classification_checkpoints.zip && rm -rf /tmp/cookies.txt &&
unzip classification_checkpoints.zip &&
rm -rf classification_checkpoints.zip &&
cd ../../
```

### Deep learning detection with FasterRCNN

```bash
cd deep_learning/detection_models &&
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1R8fxd_CdyG5ec1grobRUut8UqCKbVFdp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R8fxd_CdyG5ec1grobRUut8UqCKbVFdp" -O detection_checkpoints.zip && rm -rf /tmp/cookies.txt &&
unzip detection_checkpoints.zip &&
rm -rf detection_checkpoints.zip &&
cd ../../
```

### Machine Learning

Still to be added

#### Suggestion for contributers

- numpy docstring format
- flake8 lintern
- useful VSCode extensions:
  - autoDocstring
  - Python Docstring Generator
  - GitLens
