from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import os

class LinearClassifier:
    feature_block = pd.DataFrame()
    model = linear_model


    def __init__(self, features, model_type='logistic'):

        self.feature_block = features

        if model_type == 'logistic':
            self.model = linear_model.LogisticRegression(penalty='none')


    def validateClassifier(self,fit, X_test, y_test):
        y_out = fit.predict(X_test)
        out_df = pd.DataFrame({'real_y':y_test, 'predicted_y':y_out})

        acc_score = self.accuracy(out_df)
        print(out_df)
        print("The accuracy score is: ", acc_score)
        #self.getvalidationset()


    def classify(self):
        X = self.feature_block.iloc[:,1:len(self.feature_block.columns)-1]
        y = self.feature_block.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 13)
        X_train = scale(X_train)
        X_test = scale(X_test)
        fit = self.model.fit(X_train, y_train)

        return self.validateClassifier(fit, X_test, y_test)


    def accuracy(self, out_df):
        y_pred = out_df["predicted_y"]
        y_true = out_df["real_y"]
        score = accuracy_score(y_true, y_pred)
        return score

    def getvalidationset(self):
        mice_data_dir = r'/Users/lucadisse/ETH/Master/FS20/StatsLab/CSV data files for analysis'
        file_regex = r'167-(glu|eth|sal)-IG-[0-9]+_(Brain_signal|Running).csv'

        for file in os.listdir(mice_data_dir):
            mouse_data_file_path = os.path.join(mice_data_dir, file)
            print(mouse_data_file_path)

        data = pd.read_csv(mouse_data_file_path)
        print(data)
