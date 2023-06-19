from models_info import path, models_info, df, characteristics
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def find_datasets(path):
    datasets_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv') and file.find('dataset') != -1:
                datasets_list.append(os.path.join(root, file))
    return datasets_list

def make_samples(dataset):
    x = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=77, stratify=y)
    return x_train, x_test, y_train, y_test


def create_model(x_train, y_train, model_info):
    model = GridSearchCV(model_info[0], model_info[1])
    model.fit(x_train, y_train)
    name = str(model_info[0])
    return model, name


def evaluate_model(x_test, y_test, model, name, df):
    y_pred = model.predict(x_test)
    results = {
        'model': name,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1': round(f1_score(y_test, y_pred), 4)
    }
    df.concat(results, ignore_index=True)


def choose_model(df, chars):
    sorted_df = df.sort_values('f1', ascending=False)
    print(f'Для датасета с характеристиками {chars} наилучшей себя показала модель {sorted_df.iloc[0, 0].tolist()} '
          f'cо значениями метрик {sorted_df.iloc[0, 3:].tolist()}')

def load_model(model, chars):
    model_name = chars + '_model.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)

def main_logic(path, models_info, df):
    datasets = find_datasets(path)
    for dataset in datasets:
        filename = os.path.basename(dataset)
        dataset_df = pd.read_csv(dataset)
        x_train, x_test, y_train, y_test = make_samples(dataset_df)
        for item in models_info:
            model, name = create_model(x_train, y_train, item)
            evaluate_model(x_test, y_test, model, name, df)
        chars = characteristics[filename[:2]]
        choose_model(df, chars)
        load_model(model, chars)

if __name__ == '__main__':
    main_logic(path, models_info, df)

