from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from compas_explore import metrics_by_group

columns = ['African-American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other']
colors = ['r', 'g', 'b', 'y', 'c', 'm']

def select_roc_threshold(classifier, params, X, y, validation_size=0.3, tpr_threshold=0.8):
    X_train, X_valdn, y_train, y_valdn = X, X, y, y#train_test_split(X, y, test_size=validation_size, random_state=42, stratify=y)
    clf = classifier(**params)
    clf.fit(X_train, y_train)
    plt.figure()
    optimal_thres = {}
    for column, color in zip(columns, colors):
        indices = (X_valdn[column] == 1)
        # Assuming predict_proba is defined. If fails use decision_function.
        scores = clf.predict_proba(X_valdn[indices])[:,1]
        # scores = clf.decision_function(X_test)
        true_vals = y_valdn[indices]
        fpr, tpr, thr = roc_curve(true_vals, scores)
        optimal_thres[column] = get_class_threshold(fpr, tpr, thr, tpr_threshold) 
        plt.plot(fpr, tpr, color=color, label=column)
    plt.plot([0,1], [0,1], 'k--', linewidth=1)
    plt.plot([0,1], [tpr_threshold, tpr_threshold], color='limegreen', linestyle='--', linewidth=1)
    plt.legend()
    plt.savefig('figures/{}_classwise_ROC.png'.format(classifier.__name__))
    return optimal_thres

def get_class_threshold(fpr, tpr, thr, tpr_threshold):
    for tp, th in zip(tpr, thr):
        if tp >= tpr_threshold:
            return th
    print(fpr, tpr, thr, tpr_threshold)

if __name__ == "__main__":
    from compas_explore import X_train, X_test, y_train, y_test
    optimal_thres = select_roc_threshold(LogisticRegression, {'C':100}, X_train, y_train, tpr_threshold=0.79)
    metrics_by_group(LogisticRegression, {'C':100}, optimal_thres)