import json
import gzip
from collections import Counter

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
import numpy as np


def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


raw = read_json('rtvslo_train_new.json.gzip')
print(len(raw), 'inputs')
head = raw

df = pd.json_normalize(head, meta=['date', 'title', 'author', 'url'])
to_drop = ['authors', 'id', 'lead', 'category', 'topics']
df.drop(columns=to_drop, inplace=True)


def extract_topic(url):
    result = []
    for url in url.values:
        topic_pair = url[0].split('/')[3:5]
        topic = ""
        if 'sport' in topic_pair:
            topic = f"{topic_pair[0]}-{topic_pair[1]}"
        else:
            topic = f"{topic_pair[0]}"
        result.append(topic)
    return pd.DataFrame(result, columns=['topic'])


def hour_weekend(timestamps):
    timestamps = pd.to_datetime(timestamps)
    timestampsdf = pd.DataFrame(columns=['hours', 'weekend'])
    timestampsdf['hours'] = timestamps.dt.hour
    timestampsdf['weekend'] = timestamps.dt.weekday > 4
    return timestampsdf


def token_number(titles):
    lengths = [len(title.split()) for title in titles]
    title_length = pd.DataFrame(lengths, columns=['title_length'])
    return title_length


def article_length(paragraphs):
    article_lengths = [len(paragraph) for paragraph in paragraphs]
    return pd.DataFrame(article_lengths, columns=['article_length'])


def count_images(figures):
    return pd.DataFrame([len(fig) for fig in figures], columns=['images'])


def calculate_important_keywords(combined, discard):
    combined_as_list = [x
                        for xs in combined
                        for x in xs]
    keyword_counter = Counter(combined_as_list)
    keyword_df = pd.DataFrame(list(keyword_counter.values()), columns=['keywords'])
    threshold = keyword_df.quantile(discard).values[0]
    important_keywords = [key for key, value in keyword_counter.items() if value > threshold]
    return important_keywords


def keywords_filter(keywords, important_keywords):
    return [word for word in keywords if word in important_keywords]


def keyword_encoding(X, important_keywords):
    mlb = MultiLabelBinarizer(classes=important_keywords)
    result = mlb.fit_transform(X['words'])
    return pd.DataFrame(result, columns=important_keywords)


class KeywordEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, survival=0.01):
        self.survival = survival
        self.important_keywords = None

    def fit(self, X, y=None):
        print(self.survival)
        self.important_keywords = calculate_important_keywords(X['keywords'] + X['gpt_keywords'], 1.0 - self.survival)
        return self

    def transform(self, X):
        X['words'] = X['keywords'] + X['gpt_keywords']
        X['words'] = X['words'].apply(keywords_filter, args=(self.important_keywords,))
        X.drop(columns=['keywords', 'gpt_keywords'], inplace=True)
        return keyword_encoding(X, self.important_keywords)

    def get_feature_names_out(self, *args, **params):
        return self.important_keywords


url_to_topic = FunctionTransformer(extract_topic, validate=False)
time_transform = FunctionTransformer(hour_weekend, validate=False)
number_of_words = FunctionTransformer(token_number, validate=False)
length_of_paragraphs = FunctionTransformer(article_length, validate=False)
number_of_images = FunctionTransformer(count_images, validate=False)


ct = make_column_transformer((url_to_topic, ['url']),
                             (time_transform, 'date'),
                             (number_of_words, 'title'),
                             (length_of_paragraphs, 'paragraphs'),
                             (number_of_images, 'figures'),
                             remainder='passthrough',
                             verbose_feature_names_out=False
                             )
ct.set_output(transform='pandas')
df1 = ct.fit_transform(df)
# df1

numeric_attributes = ['title_length', 'article_length', 'images']
categorical_attributes = ['topic', 'hours', 'weekend']
multi_label_attributes = ['keywords', 'gpt_keywords']

scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False, feature_name_combiner='concat')
ke = KeywordEncodingTransformer(survival=0.005)

print('start transforming')
ppct = make_column_transformer(
    (ohe, categorical_attributes),
    (ke, multi_label_attributes),
    remainder='passthrough',
    verbose=False,
    verbose_feature_names_out=False
)
ppct.set_output(transform='pandas')

df2 = ppct.fit_transform(df1)
print('finish')
feature_names = [*df2.columns][:-1]
X = df2[feature_names]
y = df2['n_comments']

print('splitting')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = make_column_transformer(
    (scaler, numeric_attributes),
    remainder='passthrough'
)
pipe.set_output(transform='pandas')
X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)
print('split')


reg = Lasso(alpha=0.4, random_state=42, copy_X=False, max_iter=1000)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"{mean_absolute_error(y_test, y_pred):.5f} is the r2_score for Lasso")
y_dumb = np.ones_like(y_pred) * df2['n_comments'].mean()
print(f"{mean_absolute_error(y_test, y_dumb):.5f} is the r2_score for dumb regressor (average)")


# CROSS VALIDATION
kf = KFold(n_splits=10, shuffle=True, random_state=42)

i = 0
scoreboard = []
for train_index, test_index in kf.split(X, y):
    X_tr = X.loc[train_index]
    y_tr = y.loc[train_index]

    X_te = X.loc[test_index]
    y_te = y.loc[test_index]

    X_tr = pipe.transform(X_tr)
    X_te = pipe.transform(X_te)
    reg = Lasso(alpha=0.4, random_state=42, copy_X=False, max_iter=1000)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    scoreboard.append(r2_score(y_te, y_pred))
    print(f"{mean_absolute_error(y_te, y_pred):.5f} is the MAE for Lasso in {i}th iteration")
    i += 1

print(f"{np.mean(scoreboard):.3f} expected score on new data")

# print(reg.sparse_coef_)
#
# razlaga = dict()
# for importance, feature in sorted(zip(reg.coef_, feature_names), reverse=True):
#     razlaga[feature] = importance
#
# razlaga = dict(sorted(razlaga.items(), key=lambda x:abs(x[1]), reverse=True))

