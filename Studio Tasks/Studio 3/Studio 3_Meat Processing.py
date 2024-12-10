# Studio 3
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 

## act 1: data prep
# read data
# meat1_df = pd.read_csv("w1.csv")
# meat2_df = pd.read_csv("w2.csv")
# meat3_df = pd.read_csv("w3.csv")
# meat4_df = pd.read_csv("w4.csv")

# # merge
# combined_df = pd.concat([meat1_df, meat2_df, meat3_df, meat4_df])

# # save new csv file with all data combined
# combined_df.to_csv("combined_data.csv", index = 0)

# print(combined_df)

# # shuffled data
# shuffled_df = combined_df.sample(n=len(combined_df))
# shuffled_df = shuffled_df.reset_index(drop=True)

# shuffled_df.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\Studio 3\all_data.csv", index = False)
# print(shuffled_df)

## act 2: model training
all_data_df = pd.read_csv("combined_data.csv")

feature_cols = all_data_df.loc[:, all_data_df.columns != 'class'].columns
X = all_data_df[feature_cols]
Y = all_data_df['class']

# svm (updated)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of Splitting Train: ", accuracy_score(y_test,y_pred))

# 10 fold cross validation (updated)
# clf2 = svm.SVC()
scores = cross_val_score(clf, X, Y, cv=10)
print("Accuracy of 10 fold cross validation: ", scores)

# # act 3
# X1 = all_data_df[feature_cols]
# Y1 = all_data_df['class']
# X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=1)
# clf1 = svm.SVC(kernel='rbf')
# clf1.fit(X1_train, y1_train)
# y1_pred = clf1.predict(X1_test)

# print("Accuracy of SVM kernel: ", accuracy_score(y1_test,y1_pred))

# param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#               'kernel': ['rbf']}  

# grid = GridSearchCV(clf, param_grid, refit = True, verbose = 3)

# # fitting the model for grid search
# grid.fit(X_train, y_train) 
# # print best parameter after tuning 
# print(grid.best_params_) 

# # # print how our model looks after hyper-parameter tuning 
# print(grid.best_estimator_) 

