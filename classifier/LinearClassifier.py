from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import scale
from sklearn.metrics import balanced_accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import os
from collections import Counter

class LinearClassifier:
    feature_block = pd.DataFrame()
    model = linear_model


    def __init__(self, features, model_type='logistic', penalty = 'l2'):

        if features.selection_type == 'relevant':
            if model_type == 'logistic':
                self.model = linear_model.LogisticRegression(penalty='none', max_iter = 1000)
        else:
            if model_type == 'logistic':
                self.model = linear_model.LogisticRegression(penalty='l2', solver = 'saga', max_iter=10000)
        self.feature_block = features


    def classify(self, train_test):
        X = self.feature_block.iloc[:,0:len(self.feature_block.columns)-1]
        y = self.feature_block.iloc[:, -1]
        if train_test['split_type'] == 'ratio':
            stratifier = [i[-7:] for i in self.feature_block.index]
            test_ratio = train_test['test']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, stratify = stratifier)
        else:
            subjects = train_test['train']
            train_index = [any(str(subject) in i for subject in subjects) for i in self.feature_block.index]
            test_index = [not i for i in train_index]
            X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]
        X_train = scale(X_train)
        X_test = scale(X_test)
        self.fit = self.model.fit(X_train, y_train)
        #regularized Log. Regression
        #Bayesian Logistic Regression
        #one-vs-all / one-vs-one works
        #error correcting output codes
        # multi-class Perceptron / SVM
        # nearest neighbor, generative probabilistic models

        return self.validateClassifier(self.fit, X_test, y_test)

    def validateClassifier(self, fit, X_test, y_test):
        y_out = fit.predict(X_test)
        out_df = pd.DataFrame({'real_y': y_test, 'predicted_y': y_out})
        plot_confusion_matrix(fit, X_test, y_test, display_labels = ['eth', 'glu', 'nea', 'sal']) #, normalize='true'
        acc_score = self.accuracy(out_df)
        print(out_df)
        print("The accuracy score is: ", acc_score)
        plt.show()
        # self.getvalidationset()

    def accuracy(self, out_df):
        y_pred = out_df["predicted_y"]
        y_true = out_df["real_y"]
        score = balanced_accuracy_score(y_true, y_pred)
        return score

        data = pd.read_csv(mouse_data_file_path)
        print(data)

'''    def getvalidationset(self):
        mice_data_dir = r'/Users/lucadisse/ETH/Master/FS20/StatsLab/CSV data files for analysis'
        file_regex = r'167-(glu|eth|sal)-IG-[0-9]+_(Brain_signal|Running).csv'

        for file in os.listdir(mice_data_dir):
            mouse_data_file_path = os.path.join(mice_data_dir, file)
            print(mouse_data_file_path)'''