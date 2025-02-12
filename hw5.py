import argparse
import json
import gzip
import os
import numpy as np


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


class RTVSlo:

    def __init__(self):
        pass

    def fit(self, train_data: list):
        pass
        
    def predict(self, test_data: list) -> np.array:
        pass
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    args = parser.parse_args()

    train_data = read_json(args.train_data_path)
    test_data = read_json(args.test_data_path)

    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    main()
