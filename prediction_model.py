import json
import gzip
import re
from collections import Counter
from scipy.sparse import csr_matrix

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


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
    X['combined'] = X.apply(combine_to_single_column, axis=1)
    X['article_length'] = X['combined'].apply(count_words)
    X['article_length_squared'] = X['article_length'] ** 2
    X['combined'] = X['combined'].apply(lambda x: ' '.join(x))
    return X


def drop_unused_columns(X):
    columns_to_drop = ['url', 'authors', 'date', 'title', 'paragraphs', 'figures', 'lead', 'topics', 'keywords', 'gpt_keywords', 'id', 'category']
    return X.drop(columns_to_drop, axis=1)


raw = read_json('rtvslo_train_comp.json.gz')
test_raw = read_json('rtvslo_test_comp.json.gz')

df = pd.json_normalize(raw)
df_test = pd.json_normalize(test_raw)

X = df.drop('n_comments', axis=1)
y = df['n_comments'].apply(np.sqrt)

lr = Ridge(alpha=1)
vectorizer = TfidfVectorizer()
feature_constructor = FunctionTransformer(add_features, validate=False)
dropper = FunctionTransformer(drop_unused_columns, validate=False)

transform_pipe = make_pipeline(feature_constructor, dropper)
transform_pipe.set_output(transform='pandas')
ct = make_column_transformer(
    (vectorizer, 'combined'),
    remainder='passthrough',
)

full_pipeline = make_pipeline(transform_pipe, ct)

X_train = full_pipeline.fit_transform(X)
X_test = full_pipeline.transform(df_test)

lr.fit(X, y)
y_pred = lr.predict(X_test)
