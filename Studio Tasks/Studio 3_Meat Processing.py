# Studio 3
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

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

# shuffled_df.to_csv("all_data.csv", index = False)
# print(shuffled_df)

## act 2: model training
all_data_df = pd.read_csv("combined_data.csv")

feature_cols = all_data_df.loc[:, all_data_df.columns != 'class'].columns
X = all_data_df[feature_cols]
Y = all_data_df['class']

# svm (updated)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
clf = svm.SVC(C=10, gamma=0.0001, kernel = "rbf")
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print("Accuracy of Splitting Train: ", accuracy_score(Y_test,Y_pred))

# 10 fold cross validation (updated)
print("Accuracy of 10 fold cross validation: ", cross_val_score(clf, X, Y, cv=10))

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
grid = GridSearchCV(clf, param_grid, refit = True, verbose = 3)

# grid.fit(X_train, Y_train) 
# print(grid.best_params_) 
# print(grid.best_estimator_) 


# feature selection with 10 best features
selector = SelectKBest(f_classif, k=100)
X_feature = selector.fit_transform(X, Y)

X_train_feature, X_test_feature, Y_train_feature, Y_test_feature = train_test_split(X_feature, Y, test_size=0.3, random_state=1)

grid.fit(X_train_feature, Y_train_feature)
svm_best_features = grid.best_estimator_

# train test split + best parameters + best features
Y_pred_best_features = svm_best_features.predict(X_test_feature)
print("Accuracy of train test split validation with best SVM features: ", accuracy_score(Y_test_feature, Y_pred_best_features))

# 10 fold cross validation + best parameters + best features
print("Accuracy of 10 fold cross validation with best SVM features: ", cross_val_score(svm_best_features, X_feature, Y, cv=10))

# 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(X_pca, Y, test_size=0.3, random_state=1)

grid.fit(X_train_pca, Y_train_pca)
svm_pca = grid.best_estimator_

# train test split + best parameters + PCA
Y_pred_pca = svm_pca.predict(X_test_pca)
print("Accuracy of train test split validation with principle components: ", accuracy_score(Y_test_pca, Y_pred_pca))

# 10 fold cross validation + best parameters + PCA
print("Accuracy of 10 fold cross validation with principal components: ", cross_val_score(svm_pca, X_pca, Y, cv=10))