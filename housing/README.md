## Introduction 
This project does XXX, YYY, ZZZ

Objective

## File structure

```
|- main.py # you can write comment here
|- data/
|-- raw.csv
|-- .....csvef
|-- run.sh
```

## Exploratory Data Science
Some text

## Data Cleaning
Some text

## Modelling
Some text

## Result
Some text

# Getting Started - Instruction
## Model training
```
python -m housing.main -mode train -csv data/train_val.csv -model rf -cv
python -m housing.main -mode train -csv data/train_val.csv -model svr -cv
```

## Inferencing / predicting
```
python -m housing.main --mode predict --input_csv to_predict.csv --model random_forest --output_csv predicted.csv 
```


## References

Useful Links
- Google Python style guide https://google.github.io/styleguide/pyguide.html