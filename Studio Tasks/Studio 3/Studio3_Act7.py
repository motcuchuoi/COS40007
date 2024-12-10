# Studio3_Act7
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("all_data.csv")

#sgd, randomforest, mlp on original dataset

feature_cols = df.loc[:, df.columns != 'class'].columns
X = df[feature_cols]
Y = df['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# svm
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
clf = svm.SVC()
clf.fit(X_train, Y_train) 
Y_pred = clf.predict(X_test)

# svm train test split
print("SVM train test split accuracy: ", accuracy_score(Y_test,Y_pred))
# 10 fold cross validation
print("SVM 10 fold cross accuracy: ", cross_val_score(clf, X, Y, cv=10))

# sgd
sgd_clf = SGDClassifier(loss='hinge', random_state=1)
sgd_clf.fit(X_train, Y_train)
Y_pred_sgd = sgd_clf.predict(X_test)

# sgd train test split
print('SGD train test split accuracy: ', accuracy_score(Y_test, Y_pred_sgd))
# sgd 10 fold cross
print('SGD 10 fold cross accuracy: ', cross_val_score(sgd_clf, X, Y, cv=10))

# randomforest
rf_clf = RandomForestClassifier(random_state=1)
rf_clf.fit(X_train, Y_train)
Y_pred_rf = rf_clf.predict(X_test)

# randomforest train test split
print('RandomForest train test split accuracy: ', accuracy_score(Y_test, Y_pred_rf))
# randomforest 10 fold cross
print('RandomForest 10 fold cross accuracy: ', cross_val_score(rf_clf, X, Y, cv=10))

# mlp
mlp_clf = MLPClassifier(random_state=1, max_iter=500)
mlp_clf.fit(X_train, Y_train)
Y_pred_mlp = mlp_clf.predict(X_test)

# mlp train test split
print('MLP train test split accuracy: ', accuracy_score(Y_test, Y_pred_mlp))
# mlp 10 fold cross
print('MLP train test split accuracy: ', cross_val_score(mlp_clf, X, Y, cv=10))
