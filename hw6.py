import argparse
import json
import gzip
import os
import numpy as np
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def extract_subtopic(url):
    topic_pair = url.split('/')[3:5]
    search_subtopics = ['svet', 'sport', 'kultura', 'zabava-in-slog']
    for search in search_subtopics:
        if search in topic_pair:
            return f"{topic_pair[0]}-{topic_pair[1]}"
    return f"{topic_pair[0]}"


def combine_to_single_column(row):
    combined_list = [row['title'], row['lead']] + row['paragraphs']
    filtered_list = [re.sub(r'[^A-ZŠĐČĆŽšđčćža-z0-9\s]', '', item) for item in combined_list]
    return filtered_list


def count_words(combined_list):
    word_count = sum(len(item.split()) for item in combined_list)
    return word_count


def categorize_time_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    else:
        return 'night'


def add_features(X):
    X['n_images'] = X['figures'].apply(len)
    timestamp = pd.to_datetime(X['date'])
    X['day_time'] = timestamp.dt.hour.apply(categorize_time_of_day)
    X['weekend'] = timestamp.dt.weekday.apply(lambda x: True if x > 4 else False)
    X['subtopic'] = X['url'].apply(extract_subtopic)
    X['combined'] = X.apply(combine_to_single_column, axis=1)
    X['article_length'] = X['combined'].apply(count_words)
    X['article_length_squared'] = X['article_length'] ** 2
    X['combined'] = X['combined'].apply(lambda x: ' '.join(x))
    return X


def drop_unused_columns(X):
    columns_to_drop = ['url', 'authors', 'date', 'title', 'paragraphs', 'figures', 'lead', 'keywords', 'gpt_keywords',
                       'id']
    return X.drop(columns_to_drop, axis=1)


class RTVSlo:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.feature_constructor = FunctionTransformer(add_features, validate=False)
        self.dropper = FunctionTransformer(drop_unused_columns, validate=False)
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown='ignore')

        self.transform_pipe = make_pipeline(self.feature_constructor, self.dropper)
        self.transform_pipe.set_output(transform='pandas')
        self.ct = make_column_transformer(
            (self.vectorizer, 'combined'),
            (self.scaler, ['article_length', 'article_length_squared', 'n_images']),
            (self.ohe, ['topics', 'subtopic', 'day_time', 'weekend']),
            remainder='passthrough',
        )

        self.full_pipeline = make_pipeline(self.transform_pipe, self.ct)

        self.erf = RandomForestRegressor(criterion='squared_error', random_state=42, n_estimators=20, n_jobs=3)
        self.elr = Ridge(alpha=1)
        self.vg = VotingRegressor([('Linear_regression', self.elr), ('Random_forest', self.erf)], n_jobs=4)

    def fit(self, train_data: list):
        df = pd.json_normalize(train_data)
        X = df.drop(['n_comments', 'category'], axis=1)
        y_train = df['n_comments'].apply(np.sqrt)
        X_train = self.full_pipeline.fit_transform(X)
        self.vg.fit(X_train, y_train)
        return None
        
    def predict(self, test_data: list) -> np.array:
        df_test = pd.json_normalize(test_data)
        X_test = self.full_pipeline.transform(df_test)
        y_pred = self.vg.predict(X_test)
        y_pred[y_pred < 0] = 0
        return y_pred ** 2


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
