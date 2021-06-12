#!/bin/sh
#python -m pip install -r requirements.txt
python -m src.main -mode train -csv data/train_val.csv -app reg -model rf -cv 5