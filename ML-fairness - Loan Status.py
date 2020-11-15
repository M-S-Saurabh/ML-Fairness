#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.svm import SVC



def preprocess_data():
    train_df = pd.read_csv('Downloads/ML/Project/archive/train.csv')
    train_df.info()
    train_df = train_df.drop(columns=['Loan_ID'])
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
    print(categorical_columns)
    numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    print(numerical_columns)
    return train_df


def visualize_categorical_data(train_df):
    fig,axes = plt.subplots(4,2,figsize=(12,15))
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
    for idx,cat_col in enumerate(categorical_columns):
        row,col = idx//2,idx%2
        sns.countplot(x=cat_col,data=train_df,hue='Loan_Status',ax=axes[row,col])
    plt.subplots_adjust(hspace=1)
    plt.show()
    
    
def visualize_numerical_data(train_df):
    fig,axes = plt.subplots(1,3,figsize=(17,5))
    numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for idx,cat_col in enumerate(numerical_columns):
        sns.boxplot(y=cat_col,data=train_df,x='Loan_Status',ax=axes[idx])
    print(train_df[numerical_columns].describe())
    plt.subplots_adjust(hspace=1)
    plt.show()
    
    
def encode_data(train_df):
    train_df_encoded = pd.get_dummies(train_df,drop_first=True)
    train_df_encoded['Gender_Female'] = np.where(train_df_encoded['Gender_Male'] == 0, 1, 0)
    train_df_encoded['Married_No'] = np.where(train_df_encoded['Married_Yes'] == 0, 1, 0)
    train_df_encoded['Self_Employed_No'] = np.where(train_df_encoded['Self_Employed_Yes'] == 0, 1, 0)
    X = train_df_encoded.drop(columns='Loan_Status_Y')
    y = train_df_encoded['Loan_Status_Y']
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)
    return X_train, X_test, y_train, y_test

    
def transform_data(X_train, X_test):
    imp = SimpleImputer(strategy='mean')
    imp_train = imp.fit(X_train)
    X_train = imp_train.transform(X_train)
    X_test_imp = imp_train.transform(X_test)
    return X_train, X_test_imp


def build_model_non_thresh(classifier, params, X_train, X_test_imp, y_train, y_test):
    clf = classifier(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test_imp)
    return y_pred


def build_model(classifier, params, X_train, X_test_imp, y_train, y_test):
    train_accuracies = []
    train_f1_scores = []
    test_accuracies = []
    test_f1_scores = []
    thresholds = []
    
    clf = classifier(**params)
    clf.fit(X_train,y_train)

    for thresh in np.arange(0.1,0.9,0.1):
        y_pred_train_thresh = clf.predict_proba(X_train)[:,1]
        y_pred_train = (y_pred_train_thresh > thresh).astype(int)

        train_acc = accuracy_score(y_train,y_pred_train)
        train_f1 = f1_score(y_train,y_pred_train)

        y_pred_test_thresh = clf.predict_proba(X_test_imp)[:,1]
        y_pred_test = (y_pred_test_thresh > thresh).astype(int) 

        test_acc = accuracy_score(y_test,y_pred_test)
        test_f1 = f1_score(y_test,y_pred_test)

        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)
        test_accuracies.append(test_acc)
        test_f1_scores.append(test_f1)
        thresholds.append(thresh)
        
    return clf, train_accuracies, test_accuracies, train_f1_scores, test_f1_scores, thresholds
    

def plot_accuracies(train_accuracies, test_accuracies, train_f1_scores, test_f1_scores, thresholds):
    thresh_reg = {"Training Accuracy": train_accuracies, "Test Accuracy": test_accuracies, "Training F1": train_f1_scores, "Test F1":test_f1_scores, "Decision Threshold": thresholds }
    thresh_reg_df = pd.DataFrame.from_dict(thresh_reg)

    plot_df = thresh_reg_df.melt('Decision Threshold', var_name='Metrics', value_name="Values")
    fig,ax = plt.subplots(figsize=(15,5))
    sns.pointplot(x="Decision Threshold", y="Values",hue="Metrics", data=plot_df,ax=ax)
    plt.show()
    
    
def choose_thresholds(train_accuracies, test_accuracies, train_f1_scores, test_f1_scores, thresholds):
    for th in range(len(thresholds)-1):
        if test_f1_scores[th] > test_f1_scores[th+1]:
            return thresholds[th]
    return 0.5


def plot_confusion_matrix(th, clf, X_test_imp, y_test):
    y_pred_test_thresh = clf.predict_proba(X_test_imp)[:,1]
    y_pred = (y_pred_test_thresh > th).astype(int) 
    print("Test Accuracy: ",accuracy_score(y_test, y_pred))
    print("Test F1 Score: ",f1_score(y_test, y_pred))
    print("Confusion Matrix on Test Data")
    print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    return y_pred
    
    
def fp_by_class(X_test, y_test, y_pred, columns):
    fpr_list = []
    sample_counts = []
    for val in columns:
        class_indices = (X_test[val] == 1)
        y_test_col = y_test[class_indices]
        y_pred_col = y_pred[class_indices]
        fp = np.count_nonzero((y_pred_col == 1)  & (y_test_col == 0))
        tn = np.count_nonzero(y_test_col == 0)
        fpr = fp / (fp + tn)
        fpr_list.append(fpr)
        sample_counts.append(len(y_pred_col))
    return fpr_list, sample_counts

def tp_by_class(X_test, y_test, y_pred, columns):
    tpr_list = []
    sample_counts = []
    for val in columns:
        class_indices = (X_test[val] == 1)
        y_test_col = y_test[class_indices]
        y_pred_col = y_pred[class_indices]
        tp = np.count_nonzero((y_test_col == 1))
        fn = np.count_nonzero((y_pred_col == 0)  & (y_test_col == 1))
        tpr = tp / (tp + fn)
        tpr_list.append(tpr)
        sample_counts.append(len(y_pred_col))
    return tpr_list, sample_counts


def plot_fpr(columns, fpr_list):
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(columns)]
    plt.bar(x_pos, fpr_list, color='green')
    plt.xlabel("Factors")
    plt.ylabel("False Positive Rate")
    plt.title("FPR variation by various factors")
    plt.xticks(x_pos, columns)
    plt.show()
    
    
def plot_tpr(columns, tpr_list):
    plt.style.use('ggplot')
    x_pos = [i for i, _ in enumerate(columns)]
    plt.bar(x_pos, tpr_list, color='green')
    plt.xlabel("Factors")
    plt.ylabel("True Positive Rate")
    plt.title("TPR variation by various factors")
    plt.xticks(x_pos, columns)
    plt.show()
    
    
def plot_model_wise_fpr(X_test, y_test, y_pred, model):
    columns = [['Self_Employed_No', 'Self_Employed_Yes']]
    #['Married_Yes', 'Married_No'], ['Gender_Male', 'Gender_Female'], ['Property_Area_Semiurban', 'Property_Area_Urban'],
    for cols in columns:
        fpr_list, sample_counts = fp_by_class(X_test, y_test, y_pred, cols)
        print("\nFPR results for: " + model)
        print("----------------------------------")
        for col, count, fpr in zip(cols, sample_counts, fpr_list):
            print("{}:{:.2f} in {} samples".format(col, fpr, count))
        plot_fpr(cols, fpr_list)
        
        
def plot_model_wise_tpr(X_test, y_test, y_pred, model):
    columns = [['Self_Employed_No', 'Self_Employed_Yes']]
    #['Married_Yes', 'Married_No'], ['Gender_Male', 'Gender_Female'], ['Property_Area_Semiurban', 'Property_Area_Urban'],
    for cols in columns:
        tpr_list, sample_counts = tp_by_class(X_test, y_test, y_pred, cols)
        print("\nTPR results for: " + model)
        print("----------------------------------")
        for col, count, tpr in zip(cols, sample_counts, tpr_list):
            print("{}:{:.2f} in {} samples".format(col, tpr, count))
        plot_tpr(cols, tpr_list)
    

def demographic_parity(X_test, y_test, y_pred):
    cols = ['Self_Employed_No', 'Self_Employed_Yes']
    group_probs = []
    sample_counts = []
    for val in cols:
        # For each class
        class_indices = (X_test[val] == 1)
        y_test_col = y_test[class_indices]
        y_pred_col = y_pred[class_indices]
        nobail_count = np.count_nonzero(y_pred_col == 1)
        nobail_prob = nobail_count / len(y_pred_col)
        group_probs.append(nobail_prob)
        sample_counts.append(len(y_pred_col))
    return group_probs, sample_counts


def plot_group_results(columns, values, sample_counts, clf_name, 
                metricname=None, title=None,
                metricabbrev=None, xlabel='cols', color='green'):
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
        plt.title("{} variation by cols".format(metricabbrev, xlabel))
    else:
        plt.title(title)
    plt.xticks(x_pos, columns)
    plt.show()
    
    
def main():
    train_df = preprocess_data()
    visualize_categorical_data(train_df)
    visualize_numerical_data(train_df)
    X, y = encode_data(train_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test_imp = transform_data(X_train, X_test)
    
    #Without having threshold values - Logistic Regression
#     log_y_non_thres_pred = build_model_non_thresh(LogisticRegression, {'solver':'liblinear'}, X_train, X_test_imp, y_train, y_test)
#     plot_model_wise_fpr(X_test, y_test, log_y_non_thres_pred, 'Non Thresh Logistic Regression')
    
    #Without having threshold values - SVC
#     svc_y_non_thres_pred = build_model_non_thresh(SVC, {'kernel':'rbf'}, X_train, X_test_imp, y_train, y_test)
#     plot_model_wise_fpr(X_test, y_test, svc_y_non_thres_pred, ' Non Thresh SVC')
    
    #Logistic Regression
    log_clf, log_train_accuracies, log_test_accuracies, log_train_f1_scores, log_test_f1_scores, log_thresholds = build_model(LogisticRegression, {'solver':'liblinear'}, X_train, X_test_imp, y_train, y_test)
    plot_accuracies(log_train_accuracies, log_test_accuracies, log_train_f1_scores, log_test_f1_scores, log_thresholds)
    log_th = choose_thresholds(log_train_accuracies, log_test_accuracies, log_train_f1_scores, log_test_f1_scores, log_thresholds)
    print(log_th)
    log_y_pred = plot_confusion_matrix(0.3, log_clf, X_test_imp, y_test)
    
    #SVM
    svc_clf, svc_train_accuracies, svc_test_accuracies, svc_train_f1_scores, svc_test_f1_scores, svc_thresholds = build_model(SVC, {'kernel':'rbf', 'probability':True}, X_train, X_test_imp, y_train, y_test)
    plot_accuracies(svc_train_accuracies, svc_test_accuracies, svc_train_f1_scores, svc_test_f1_scores, svc_thresholds)
    svc_th = choose_thresholds(svc_train_accuracies, svc_test_accuracies, svc_train_f1_scores, svc_test_f1_scores, svc_thresholds)
    print(svc_th)
    svc_y_pred = plot_confusion_matrix(svc_th, svc_clf, X_test_imp, y_test)
    
    plot_model_wise_fpr(X_test, y_test, log_y_pred, 'Logistic Regression')
    plot_model_wise_tpr(X_test, y_test, log_y_pred, 'Logistic Regression')
    
    group_probs, sample_counts = demographic_parity(X_test, y_test, log_y_pred)
    cols = ['Self_Employed_No', 'Self_Employed_Yes']
    plot_group_results(cols, group_probs, sample_counts, 'Logistic Regression', 
                    metricname='Loan Granting Probability', metricabbrev='Parity',
                    title='Demographic Parity', color='red')
#     plot_model_wise_fpr(X_test, y_test, svc_y_pred, 'SVC')
    
    print("end")


if __name__ == "__main__":
    main()

