import argparse
import logging

from numpy.core.arrayprint import str_format
from . import process_data
from . import model
from . import utils
import joblib

# MODE - TRAIN
# df_train = split it into train and valid set
# during experimentation, only play with train and validation set

# MODE - TEST
# df_test = treat as if this is hidden test set
# only clean it when in predict mode
# never use test in train mode


def main():
    """
    argparse - https://realpython.com/command-line-interfaces-python-argparse/
    logging - https://realpython.com/python-logging/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mode", help="Select mode of operation", choices=["train", "predict"])

    parser.add_argument("-csv", help="path to csv file")

    parser.add_argument(
        "-app", choices=["reg", "cla"], help="select approach: regression or classification")

    parser.add_argument(
        "-cv", nargs='?', help="Num. of folds for cross validation", const=5, type=int)

    parser.add_argument("-model", nargs='?',
                        help="Which model to use", const='rf', type=str)

    args = parser.parse_args()

    # print('args.mode ', args.mode)  # train, predict
    # print('args.csv ', args.csv)  # ./data/train.csv
    # print('args.cv ', args.cv)  # default 5
    # print('args.model ', args.model)  # rf / xgb

    if args.mode == "train" and args.csv != None:
        logging.info("Running %s Model...", args.model)
        logging.info("Running cross-validation %s folds", args.cv)
        df_raw = process_data.read_data(args.csv)  # train.csv
        # df_cleaned = process_data.clean_data(df_raw)
        df_cleaned = process_data.clean_data(df_raw, args.mode)

        if args.app == "reg":
            mae = model.train(df_cleaned, app=args.app,
                              model=args.model, cv=args.cv)
            logging.info("mean_absolute_error = %s", mae)
        elif args.app == "cla":
            df_cleaned_bin = process_data.get_bin(df_cleaned)
            mean_accuracy, std_accuracy = model.train(df_cleaned_bin, app=args.app,
                                                      model=args.model, cv=args.cv)
            logging.info("mean accuracy = %s", mean_accuracy)
            logging.info("std accuracy = %s", std_accuracy)

    elif args.mode == "predict" and args.csv != None:

        logging.info("Running XGB Model...")
        df_raw = process_data.read_data(args.csv)  # train.csv
        df_cleaned = process_data.clean_data(df_raw, args.mode)
        # df_cleaned = process_data.clean_data(df_raw)
        if args.app == "reg":
            # mae = model.train(df_cleaned, model=args.model, cv=args.cv)
            joblib_file = "./model/joblib_xgb_model.pkl"
            model.predict(df_cleaned, joblib_file)
        elif args.app == "cla":
            #df_cleaned_bin = process_data.get_bin(df_cleaned)
            joblib_file = "./model/joblib_xgb_cla_model.pkl"
            model.predict(df_cleaned, joblib_file)

            # elif args.mode == "predict":
            #     if args.csv != None and args.model != None:
            #         #logging.debug("inside predict mode for unseen test dataset")
            #         df_raw = process_data.read_data(
            #             args.csv)  # reading unseen test set
            #         df_cleaned = process_data.clean_data(df_raw)
            #         df_pred = model.test(df_cleaned, args.model)
            #         df_pred.to_csv('./data/test_predictions.csv')
            # else:
            #     raise ValueError("Enter mode as train or predict")

            # parser = argparse.ArgumentParser()
            # # df_cleaned = clean_data(file="xxx.csv")
            # # df_encode_normalised = encode_normalise_data(df_cleaned)
            # # # split is the train test split
            # df = rf_reg(df_cleaned, cv=False, split=0.2)


if __name__ == "__main__":
    main()
