## Introduction 
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this project aims to predict the final price of each home.

Project aims to predict house prices in two ways
1. as a regression problem
2. as a classification problem (by binning the prices into 4 categorical quartiles)

## File structure

```
|-- src
    |-- main.py
    |-- model.py
    |-- process_data.py
|-- data
    |-- train_val.csv
    |-- test.csv 
|-- model
    |-- joblib_xgb_cla_model.pkl
    |-- joblib_xgb_model.pkl
    |-- processed_columns.txt
|-- prediction
    |-- prediction.csv
|-- README.md
|-- run.sh
|-- requirements.txt
|-- EDA.ipynb
```

## Exploratory Data Science
Some text

## Data Cleaning

## Modelling
Some text

## Result
Some text

# Getting Started - Instruction
## Model training
### Regression with Random Forest or XGB
```
python -m src.main -mode train -csv data/train_val.csv -app reg -model rf 
python -m src.main -mode train -csv data/train_val.csv -app reg -model xgb
```
### Classification with Random Forest or XGB
```
python -m src.main -mode train -csv data/train_val.csv -app cla -model rf 
python -m src.main -mode train -csv data/train_val.csv -app cla -model xgb
```

## Inferencing / Predicting
### Prediction outcome: Regression
```
python -m src.main -mode predict -csv data/test.csv -app reg
```
### Prediction outcome: Classification
```
python -m src.main -mode predict -csv data/test.csv -app cla
```

# Output
CSV file generated after running predict mode. 

Kaggle competition submission friendly. 
```
|-- prediction
    |-- prediction.csv
```

## References
Useful Links
- Google Python style guide https://google.github.io/styleguide/pyguide.html
- Kaggle source https://www.kaggle.com/c/home-data-for-ml-course