import argparse


from data.data import process_raw_dataset
from model.model import train, predict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('function', choices=['process', 'train', 'predict'])
    args = parser.parse_args()

    if args.function == 'process':
        process_raw_dataset()

    elif args.function == 'train':
        train()

    elif args.function == 'predict':
        predict()
