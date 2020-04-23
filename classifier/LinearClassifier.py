from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import scale
from sklearn.metrics import balanced_accuracy_score, plot_confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
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

        X_train = scale(X_train)
        X_test = scale(X_test)

        #Handle class imbalance by assigning weights to the samples in the loss function
        sample_weights = compute_sample_weight('balanced', y_train)

        self.fit = self.model.fit(X_train, y_train, sample_weight=sample_weights)

        #regularized Log. Regression
        #Bayesian Logistic Regression
        #one-vs-all / one-vs-one works
        #error correcting output codes
        # multi-class Perceptron / SVM
        # nearest neighbor, generative probabilistic models

        return self.validateClassifier(X_test, y_test)


    def validateClassifier(self, X_test, y_test):
        y_out = self.fit.predict(X_test)
        y_out_confidence = self.fit.decision_function(X_test)
        y_out_confidence = 1/(1+np.exp(-y_out_confidence))

        plot_confusion_matrix(self.fit, X_test, y_test, display_labels = ['eth', 'glu', 'nea', 'sal']) #, normalize='true'
        plt.show()

        #check if we do binary classification or multilabel
        if y_out_confidence.shape[1] is 1:
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
        return self.fit.predict(X_test)
