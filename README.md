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

## Data Processing
### Drop features that are 
1. dominated by single value

```
BsmtFinSF2, LowQualFinSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal, BsmtHalfBath, KitchenAbvGr, Street, Utilities, Condition2, RoofMatl, Heating, Functional, GarageQual, GarageCond 
```
2. high correlations with other features
```
GarageYrBlt, 1stFlrSF, TotRmsAbvGrd, GarageArea
```

3. large amount of missing values
```
PoolQC, MiscFeature, Alley, Fence, FireplaceQu
```

### Create new features
1. This creates a new feature which sum the lot space.
```
df['LotTotalArea'] = df['LotFrontage'] + df['LotArea']
```
2. This creates a new feature which sum the basement space.
```
df['BsmtTotalArea'] = df['TotalBsmtSF'] + df['BsmtUnfSF'] + df['BsmtFinSF1']
```
3. This creates a new feature which of total baths in the house.
```
df['TotalBathAbvGr'] = df['FullBath'] + df['HalfBath']
```

### Impute missing values
1. For numerical features, missing values are imputed with mode value.

2. For categorical features, missing values are imputed with "NA".

### Encode categorical features
1. Ordinal features are encoded to maintain the ranking integrity.

```
ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, BsmtFinType1, BsmtFinType2, BsmtExposure
```

2. Other nominal features are encoded via one-hot encoding.  

### Binning the features
For classification approach, the SalePrice are binned into 4, based on the quartiles. 

0: min - 25%

1: 25% - 50%

2: 50% - 75%

3: 75% - max


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