import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def group_credit_hist(x):
    if x in ['A30', 'A31', 'A32']:
        return 'None/Paid'
    elif x == 'A33':
        return 'Delay'
    elif x == 'A34':
        return 'Other'
    else:
        return 'NA'


def group_employ(x):
    if x == 'A71':
        return 'Unemployed'
    elif x in ['A72', 'A73']:
        return '1-4 years'
    elif x in ['A74', 'A75']:
        return '4+ years'
    else:
        return 'NA'


def group_savings(x):
    if x in ['A61', 'A62']:
        return '<500'
    elif x in ['A63', 'A64']:
        return '500+'
    elif x == 'A65':
        return 'Unknown/None'
    else:
        return 'NA'


def group_status(x):
    if x in ['A11', 'A12']:
        return '<200'
    elif x in ['A13']:
        return '200+'
    elif x == 'A14':
        return 'None'
    else:
        return 'NA'


def replace_with_onehot(df, col, prefix=None):
    onehot = pd.get_dummies(df[col], prefix=prefix)
    df = df.drop(col, axis=1)
    return df.join(onehot)


def pre_process():
    column_names = ['status', 'month', 'credit_history',
                'purpose', 'credit_amount', 'savings', 'employment',
                'investment_as_income_percentage', 'personal_status',
                'other_debtors', 'residence_since', 'property', 'age',
                'installment_plans', 'housing', 'number_of_credits',
                'skill_level', 'people_liable_for', 'telephone',
                'foreign_worker', 'credit']
    ddf = pd.read_csv('data/german.data', sep=' ', header=None, names=column_names)
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                      'A92': 'female', 'A95': 'female'}
    ddf['sex'] = ddf['personal_status'].replace(status_map)

    ddf['credit_history'] = ddf['credit_history'].apply(lambda x: group_credit_hist(x))
    ddf['savings'] = ddf['savings'].apply(lambda x: group_savings(x))
    ddf['employment'] = ddf['employment'].apply(lambda x: group_employ(x))
    ddf['age'] = ddf['age'].apply(lambda x: np.float(x >= 26))
    ddf['status'] = ddf['status'].apply(lambda x: group_status(x))
    ddf.drop(ddf.columns.difference(['credit_history', 'savings', 'employment', 'sex', 'age', 'credit']), 1,
             inplace=True)

    use_cols = ['credit_history', 'savings', 'employment', 'sex', 'age', 'credit']

    for col in use_cols:
        if col == 'sex':
            ddf = replace_with_onehot(ddf, col)
        if col == 'credit_history':
            ddf = replace_with_onehot(ddf, col, prefix=col)
        if col == 'savings':
            ddf = replace_with_onehot(ddf, col, prefix=col)
        if col == 'employment':
            ddf = replace_with_onehot(ddf, col, prefix=col)

    return ddf


df = pre_process()
X = df.drop('credit', axis=1)
y = df['credit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def plot_group_results(columns, values, sample_counts, clf_name,
                metricname=None, title=None,
                metricabbrev=None, xlabel='Sex', color='green'):
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
        plt.title("{} variation by sex".format(metricabbrev, xlabel))
    else:
        plt.title(title)
    plt.xticks(x_pos, columns)
    plt.show()
    # plt.savefig("graphs/{}_{}_plot.png".format(metricabbrev, clf_name))


def fptp_by_class(X_test, y_test, y_pred, columns):
    fpr_list = []
    tpr_list = []
    sample_counts = []
    for val in columns:
        # For each class
        class_indices = (X_test[val] == 1)
        y_test_col = y_test[class_indices]
        y_pred_col = y_pred[class_indices]
        fp = np.count_nonzero((y_pred_col == 1)  & (y_test_col == 2))
        tn = np.count_nonzero((y_pred_col == 2)  & (y_test_col == 2))
        tp = np.count_nonzero((y_pred_col == 1)  & (y_test_col == 1))
        fn = np.count_nonzero((y_pred_col == 2)  & (y_test_col == 1))
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
        np.put(y_pred, np.where(indices), preds)
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
    columns = ['male', 'female']

    # Demographic parity
    group_probs, sample_counts = demographic_parity(X_test, y_test, y_pred, columns)
    plot_group_results( columns, group_probs, sample_counts, classifier.__name__,
                        metricname='Loan denial probability', metricabbrev='Parity',
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