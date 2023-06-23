import pandas as pd
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# определение пути проекта
path = '../../anomalies-detection-project'

# определение параметров моделей
models_info = (
    (
        svm.SVC(probability=True),
        {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly', 'linear']}
    ),
    (
        KNeighborsClassifier(),
        {'n_neighbors': [1, 2, 3, 5, 7, 10], 'algorithm': ['auto', 'ball_tree', 'kd_tree']}
    ),
    (
        RandomForestClassifier(),
        {'n_estimators': [100, 200], 'criterion': ['gini'],
         'min_samples_leaf': [0.1, 0.2, 1], 'min_samples_split': [0.5, 1.0, 4],
         'max_features': ['sqrt']}
    ),
    (
        GaussianNB(),
        {'var_smoothing': [0.1, 0.01, 0.001, 0.0001]}
    ),
    (
        AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion='gini', max_depth=5)),
        {'n_estimators': [50, 100], 'learning_rate': [0.1,  1.0], 'algorithm': ['SAMME.R', 'SAMME']}
    ),
    (
        BaggingClassifier(estimator=DecisionTreeClassifier(criterion='gini', max_depth=5)),
        {'n_estimators': [50, 100], 'random_state': [13, 42, 100], 'max_samples': [0.5, 0.75, 1.0]}
    ),
    (
        GradientBoostingClassifier(),
        {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0], 'max_depth': [3, 6, 9],
         'random_state': [13, 42, 100]}
    ),
)

# определение датафрейма, содержащего результаты оценивания моделей
df = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'prediction time'])

#определение характеристик
characteristics = {'nc': 'сети', 'dc': 'домена'}