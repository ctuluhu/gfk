import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import config


def main():
    np.random.seed(config.SEED)

    df_train, df_test = load_data()
    df_train, df_test = preprocess_data(df_train), preprocess_data(df_test)

    df_train = drop_selected_columns(df_train, ['v1', 'v5', 'v12_g', 'v12_p', 'v12_s', 'v15', 'v17'])
    df_test = drop_selected_columns(df_test, ['v1', 'v5', 'v12_g', 'v12_p', 'v12_s', 'v15', 'v17'])

    df_train, df_test = remove_uncommon_features(df_train, df_test)

    x_train, x_valid, y_train, y_valid = train_test_split(
        df_train.drop('classLabel', axis=1), df_train['classLabel'], test_size=0.1, random_state=config.SEED)

    x_test, y_test = df_test.drop('classLabel', axis=1), df_test['classLabel']

    x_train, x_valid, x_test = rescale_data(x_train, x_valid, x_test)

    params = config.LEARNING_PARAMS

    model = xgb.XGBClassifier(**params)
    model.fit(x_train, y_train, eval_metric=['logloss'], eval_set=[(x_train, y_train), (x_valid, y_valid)])

    results = model.evals_result()
    plt.plot(results['validation_0']['logloss'], label='train')
    plt.plot(results['validation_1']['logloss'], label='valid')
    plt.legend()
    plt.savefig('learning_curve.png')

    y_predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predictions)
    print(f'\nAccuracy: {accuracy:.2f}\n')
    print('Confusion matrix:')
    c_matrix = confusion_matrix(y_test, y_predictions)
    print(c_matrix)


def drop_selected_columns(df, columns_to_drop):
    for column in list(columns_to_drop):
        df = df.drop(column, axis=1)
    return df


def fine_tune_xgb_classifier(x, y):
    model = xgb.XGBClassifier()

    param_grid = {
        'learning_rate': [0.1, 0.5, 0.8],
        'n_estimators': [1000, 2000, 5000],
        'max_depth': [5, 6, 7],
        'min_child_weight': [3, 4, 5],
        'scale_pos_weight': [0.08],
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x, y)
    best_model = grid_search.best_estimator_

    return best_model


def load_data():
    df_train = pd.read_csv(config.TRAIN_DATASET_PATH, delimiter=';')
    df_test = pd.read_csv(config.TEST_DATASET_PATH, delimiter=';')
    return df_train, df_test


def preprocess_data(df):
    label_encoder = LabelEncoder()

    df['v1'] = label_encoder.fit_transform(df['v1'])
    df['v8'] = label_encoder.fit_transform(df['v8'])
    df['v9'] = label_encoder.fit_transform(df['v9'])
    df['v11'] = label_encoder.fit_transform(df['v11'])
    df['v16'] = label_encoder.fit_transform(df['v16'])
    df['classLabel'] = label_encoder.fit_transform(df['classLabel'])

    df['v14'] = df['v14'].astype(float)

    df['v2'] = df['v2'].str.replace(',', '.').astype(float)
    df['v3'] = df['v3'].str.replace(',', '.').astype(float)
    df['v5'] = df['v5'].str.replace(',', '.').astype(float)
    df['v6'] = df['v6'].str.replace(',', '.').astype(float)
    df['v7'] = df['v7'].str.replace(',', '.').astype(float)
    df['v15'] = df['v15'].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)

    df = pd.concat([df, pd.get_dummies(df['v4'], prefix='v4')], axis=1)
    df = df.drop(['v4'], axis=1)

    df = pd.concat([df, pd.get_dummies(df['v12'], prefix='v12')], axis=1)
    df = df.drop(['v12'], axis=1)

    return df


def rescale_data(x_train, x_valid, x_test):
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    return x_train, x_valid, x_test


def remove_uncommon_features(df_train, df_test):
    common_features = df_train.columns.intersection(df_test.columns)
    df_train = df_train[common_features]
    df_test = df_test[common_features]
    return df_train, df_test


if __name__ == '__main__':
    main()
