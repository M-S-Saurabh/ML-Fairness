import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

PATH_COMPAS_VIOLENT = os.path.join(os.getcwd(), 'data', 'compas-scores-two-years-violent.csv')

TARGET_NAME = 'is_violent_recid' # 'is_recid', 'two_year_recid'

columns = [('sex', str), ('age', int), ('race', str), ('priors_count', int), 
            ('juv_fel_count', int), ('juv_other_count', int), ('c_charge_degree', str),
            (TARGET_NAME, bool)]

usecols = [a for a,b in columns]
# types = [b for a,b in columns]

def replace_with_onehot(df, col, prefix=None):
    onehot = pd.get_dummies(df[col], prefix=prefix)
    df = df.drop(col, axis=1)
    return df.join(onehot)

def preprocess_compas(df, usecols):
    # One-hot encoding for categorical variables.
    for col in usecols:
        if col.endswith('charge_degree'):
            df = replace_with_onehot(df, col, prefix=col)
        if col == 'race':
            df = replace_with_onehot(df, col)
        if col == 'sex':
            df = replace_with_onehot(df, col)
    return df

df = pd.read_csv(PATH_COMPAS_VIOLENT,delimiter=',', usecols=usecols)
df = preprocess_compas(df, usecols)

X = df.drop(TARGET_NAME, axis=1)
y = df[TARGET_NAME]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 0.1, 1, 10, 100, 1000], 'C': [1e-3, 1e-2, 0.1, 1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1e-3, 1e-2, 0.1, 1, 10, 100, 1000]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring=score, verbose=1, n_jobs=5, cv=3
    )
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# data = np.genfromtxt(PATH_COMPAS_VIOLENT, delimiter=',', names=True, usecols=usecols, dtype=types)

# print("Violent same as is_recid?", np.count_nonzero(data['is_violent_recid'] != data['is_recid']))
# print("2-year same as is_recid?", np.count_nonzero(data['two_year_recid'] != data['is_recid']))
