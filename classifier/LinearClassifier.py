from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, plot_confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import preprocessing
from sklearn.svm import SVC
import os
from collections import Counter

class LinearClassifier:
    feature_block = pd.DataFrame()
    model = linear_model


    def __init__(self, features, model_type='logistic', penalty = 'l2', C_val=1):

        #if model_type == 'logistic':
        #   self.model = linear_model.LogisticRegression(penalty='none', max_iter = 1000)

        if model_type == 'logistic':
            self.model = linear_model.LogisticRegression(penalty= penalty, solver = 'saga', max_iter=10000, C=C_val)

        self.model=SVC(gamma='auto')
        self.feature_block = features


    #TODO should be called trainClassifier
    def classify(self, train_test):
        X = self.feature_block.drop(columns='target_class')
        y = self.feature_block['target_class']

        if train_test['split_type'] == 'ratio':
            #stratifier = [i[-7:] for i in self.feature_block.index]
            test_ratio = train_test['test']
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, stratify = stratifier)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, stratify=y)
        else:
            subjects = train_test['train']
            #TODO write out for loop to inlude checksums, sometimes items of subsjects not in feature_block.index but no error raised
            train_index = [any(str(subject) in i for subject in subjects) for i in self.feature_block.index]
            test_index = [not i for i in train_index]

            X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        #Handle class imbalance by assigning weights to the samples in the loss function
        sample_weights = compute_sample_weight('balanced', y_train)

        self.fit = self.model.fit(X_train, y_train, sample_weight=sample_weights)

        #regularized Log. Regression
        #Bayesian Logistic Regression
        #one-vs-all / one-vs-one works
        #error correcting output codes
        # multi-class Perceptron / SVM
        # nearest neighbor, generative probabilistic models

        return self.validateClassifier(X_test, y_test, plot = True)

    def k_fold_train_classifier(self):
        tot_test_sum, tot_test_samples, tot_conf_matr = 0, 0, None
        X = self.feature_block.drop(columns='target_class')
        y = self.feature_block['target_class']

        skf = StratifiedKFold(n_splits=20)
        for train_index, test_index in skf.split(X, y):
            X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            sample_weights = compute_sample_weight('balanced', y_train)

            self.fit = self.model.fit(X_train, y_train, sample_weight=sample_weights)
            acc, conf_matr = self.validateClassifier(X_test, y_test)

            tot_test_sum += acc*len(test_index)
            tot_test_samples += len(test_index)

            if tot_conf_matr is None:
                tot_conf_matr = conf_matr
            else:
                tot_conf_matr += conf_matr

        return tot_test_sum/tot_test_samples, tot_conf_matr

    def validateClassifier(self, X_test, y_test, plot = False):
        y_out = self.fit.predict(X_test)
        y_out_confidence = self.fit.decision_function(X_test)
        y_out_confidence = 1/(1+np.exp(-y_out_confidence))

        if plot:
            plot_confusion_matrix(self.fit, X_test, y_test, display_labels = ['eth', 'glu', 'nea', 'sal']) #, normalize='true'
            plt.show()

        #check if we do binary classification or multilabel
        if len(y_out_confidence.shape) is 1:
            out_df = pd.DataFrame({'real_y': y_test, 'predicted_y': y_out, 'predicted_y_confidence': y_out_confidence})
            acc_score, auc_score = self.accuracy(out_df)
            print(out_df)
            print("The accuracy score is: ", acc_score)
            print("The AUC score is: ", auc_score)
        else:
            out_df = pd.DataFrame({'real_y': y_test, 'predicted_y': y_out})
            acc_score  = self.accuracy(out_df)
            print(out_df)
            print("The accuracy score is: ", acc_score)

            return acc_score, confusion_matrix(y_test, y_out, labels = ['eth', 'glu', 'nea', 'sal'])

    def accuracy(self, out_df):
        y_pred = out_df["predicted_y"]
        y_true = out_df["real_y"]
        accuracy = balanced_accuracy_score(y_true, y_pred)

        if "predicted_y_confidence" in out_df:
            y_pred_confidence = out_df["predicted_y_confidence"]
            auc_score = roc_auc_score(y_true, y_pred_confidence)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_confidence, pos_label=2)
            return accuracy, auc_score

        # if multilable classification return only accuracy
        return accuracy


    def predict(self, X):
        return self.fit.predict(X)
