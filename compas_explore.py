import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def explore_algorithm(algo, tuned_parameters, scores = ['accuracy']):
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            algo(), tuned_parameters, scoring=score, verbose=1, n_jobs=5, cv=3
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

def plot_group_results(columns, values, sample_counts, clf_name, 
                metricname=None, title=None,
                metricabbrev=None, xlabel='Race', color='green'):
    if metricabbrev is None: metricabbrev = metricname
    print("-----------------------------------------------------")
    print(" {} results for {}".format(metricname, clf_name))
    print("-----------------------------------------------------")
    for col, count, val in zip(columns, sample_counts, values):
        print("{}:{:.2f} in {} samples".format(col, val, count))

    plt.figure()
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(columns)]
    plt.bar(x_pos, values, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(metricname)
    if title is None:
        plt.title("{} variation by race".format(metricabbrev, xlabel))
    else:
        plt.title(title)
    plt.xticks(x_pos, columns)
    plt.savefig("figures/{}_{}_plot.png".format(metricabbrev, clf_name))

def fptp_by_class(X_test, y_test, y_pred, columns):
    fpr_list = []
    tpr_list = []
    sample_counts = []
    for val in columns:
        # For each class
        class_indices = (X_test[val] == 1)
        y_test_col = y_test[class_indices]
        y_pred_col = y_pred[class_indices]
        fp = np.count_nonzero((y_pred_col == 1)  & (y_test_col == 0))
        tn = np.count_nonzero((y_pred_col == 0)  & (y_test_col == 0))
        tp = np.count_nonzero((y_pred_col == 1)  & (y_test_col == 1))
        fn = np.count_nonzero((y_pred_col == 0)  & (y_test_col == 1))
        fpr = fp / (fp+tn) if (fp+tn > 0) else 0
        tpr = tp / (tp+fn) if (tp+fn > 0) else 0
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        sample_counts.append(len(y_pred_col))
    return fpr_list, tpr_list, sample_counts

def demographic_parity(X_test, y_test, y_pred, columns):
    group_probs = []
    sample_counts = []
    for val in columns:
        # For each class
        class_indices = (X_test[val] == 1)
        y_test_col = y_test[class_indices]
        y_pred_col = y_pred[class_indices]
        nobail_count = np.count_nonzero(y_pred_col == 1)
        nobail_prob = nobail_count / len(y_pred_col)
        group_probs.append(nobail_prob)
        sample_counts.append(len(y_pred_col))
    return group_probs, sample_counts

def threshold_predict(model, X, thresholds):
    y_pred = np.zeros(X.shape[0])
    for column, threshold in thresholds.items():
        indices = (X[column] == 1)
        if threshold is None: threshold = 0.5
        preds = (model.predict_proba(X[indices])[:,1] >= threshold).astype(int)
        np.put( y_pred, np.where(indices), preds)
    return y_pred

def metrics_by_group(classifier, params, thresholds=None):
    clf = classifier(**params)
    clf.fit(X_train, y_train)
    print()
    print(clf)
    if thresholds is None:
        y_pred = clf.predict(X_test)
    else:
        y_pred = threshold_predict(clf, X_test, thresholds)
    columns = ['African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other']

    # Demographic parity
    group_probs, sample_counts = demographic_parity(X_test, y_test, y_pred, columns)
    plot_group_results( columns, group_probs, sample_counts, classifier.__name__, 
                        metricname='Bail denial probability', metricabbrev='Parity',
                        title='Demographic Parity', color='red')
    # FPR and TPR (equal opportunity)
    fpr_list, tpr_list, sample_counts = fptp_by_class(X_test, y_test, y_pred, columns)
    plot_group_results( columns, fpr_list, sample_counts, classifier.__name__,
                        metricabbrev='FPR', metricname='False positive rate', color='green')
    plot_group_results( columns, tpr_list, sample_counts, classifier.__name__, 
                        metricabbrev='TPR', metricname='True positive rate', color='blue')

    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    metrics_by_group(LogisticRegression, {'C':100})
    metrics_by_group(SVC, {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'})

# SVM_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-2, 0.1, 1, 10, 100, 1000], 'C': [1e-3, 1e-2, 0.1, 1, 10, 100, 1000]},
#                         {'kernel': ['linear'], 'C': [1e-3, 1e-2, 0.1, 1, 10, 100, 1000]}]
# # explore_algorithm(SVC, SVM_parameters)

# LR_parameters = [{'C':[1e-3, 1e-2, 0.1, 1, 10, 100, 1000]}]
# explore_algorithm(LogisticRegression, LR_parameters)