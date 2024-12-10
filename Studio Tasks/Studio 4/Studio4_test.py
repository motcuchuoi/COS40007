import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# using original data
df = pd.read_csv("combined_data.csv")

feature_cols = df.loc[:, df.columns != 'class']

X = feature_cols
Y = df['class']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size= 0.3, random_state= 1)

# SVM
clf = svm.SVC()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
# print("SVM train test split accuracy: ", accuracy_score(Y_test,Y_pred))

# SGD
sgd_clf = SGDClassifier(loss = 'hinge', random_state=1)
sgd_clf.fit(X_train, Y_train)
Y_pred_sgd = sgd_clf.predict(X_test)
# print('SGD train test split accuracy: ', accuracy_score(Y_test, Y_pred_sgd))

# randomforest
rf_clf = RandomForestClassifier(random_state=1)
rf_clf.fit(X_train, Y_train)
Y_pred_rf = rf_clf.predict(X_test)
# print('RandomForest train test split accuracy: ', accuracy_score(Y_test, Y_pred_rf))


# mlp
mlp_clf = MLPClassifier(random_state=1, max_iter=500)
mlp_clf.fit(X_train, Y_train)
Y_pred_mlp = mlp_clf.predict(X_test)
# print('MLP train test split accuracy: ', accuracy_score(Y_test, Y_pred_mlp))

# model_Y_pred = {
#     'SVM': Y_pred,
#     'SGD': Y_pred_sgd,
#     'RandomForest': Y_pred_rf,
#     'MLP': Y_pred_mlp
# }

target_names = ['0 - idle', '1 - cutting', '2 - sharpening']

print(classification_report(Y_test, Y_pred, target_names=target_names))
print(classification_report(Y_test, Y_pred_sgd, target_names=target_names))
print(classification_report(Y_test, Y_pred_rf, target_names=target_names))
print(classification_report(Y_test, Y_pred_mlp, target_names=target_names))
