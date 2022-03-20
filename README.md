# MAIA - AIA-ML-DL - Calcification Detection

- ### Cortina Uribe Alejandro
- ### Seia Joaquin Oscar
- ### Zalevskyi Vladimir

## Environment set up

Start by creating a new conda environment

``` conda update -n base -c defaults conda ```

``` conda create -n calc_det anaconda ``` 

``` conda activate calc_det ```

> TODO: Define a lighter version of the environment (not using conda create -n bla *anaconda*)


Install requirements:

``` pip install -r requirements.txt```

## To get the database there are two options:

### Option A
1. Download the raw database from the original [Drive](https://drive.google.com/file/d/19n-p9p9C0eCQA1ybm6wkMo-bbeccT_62/view) from the authors.

2. Extract the zipped images in a "data" directory inside this same directory, the directory structure should be the following:

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

3. Run the following command:


    ``` python preprocess.py -arg1 bla ...```

### Option B

Use the tracked version of the dataset with DVC following the next steps:

1. 
2. 

### Coments for developers:
    I suggest using:
        numpy docstring format
        flake8 lintern
    I suggest this extensions for VSCode:
        autoDocstring - Python Docstring Generator
        GitLens
