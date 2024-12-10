import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

df = pd.read_csv('vegemite_selected_features.csv')

feature_cols = df.loc[:, df.columns != 'Class']
X = feature_cols
Y = df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test

# Decision Tree Classifier
dtc_clf = DecisionTreeClassifier() 
dtc_clf = dtc_clf.fit(X_train,Y_train)
dtc_Y_pred = dtc_clf.predict(X_test)
# print("Decision Tree Classifier accuracy:", accuracy_score(Y_test, dtc_Y_pred))

dtc_cm = confusion_matrix(Y_test, dtc_Y_pred, labels = dtc_clf.classes_)
dtc_disp = ConfusionMatrixDisplay(confusion_matrix = dtc_cm,
                              display_labels = dtc_clf.classes_)

# SVM
svm_clf = svm.SVC()
svm_clf = svm_clf.fit(X_train, Y_train)
svm_Y_pred = svm_clf.predict(X_test)
# print("SVM accuracy:", accuracy_score(Y_test, svm_Y_pred))

svm_cm = confusion_matrix(Y_test, svm_Y_pred, labels = svm_clf.classes_)
svm_disp = ConfusionMatrixDisplay(confusion_matrix = svm_cm,
                              display_labels = svm_clf.classes_)
# SGD
sgd_clf = SGDClassifier(loss='hinge')
sgd_clf = sgd_clf.fit(X_train, Y_train)
sgd_Y_pred = sgd_clf.predict(X_test)
# print("SGD accuracy:", accuracy_score(Y_test, sgd_Y_pred))

sgd_cm = confusion_matrix(Y_test, sgd_Y_pred, labels = sgd_clf.classes_)
sgd_disp = ConfusionMatrixDisplay(confusion_matrix = sgd_cm,
                              display_labels = sgd_clf.classes_)
# RandomForest
rf_clf = RandomForestClassifier(random_state=1)
rf_clf = rf_clf.fit(X_train, Y_train)
rf_Y_pred = rf_clf.predict(X_test)
# print("Random Forest accuracy:", accuracy_score(Y_test, rf_Y_pred))

rf_cm = confusion_matrix(Y_test, rf_Y_pred, labels = rf_clf.classes_)
rf_disp = ConfusionMatrixDisplay(confusion_matrix = rf_cm,
                              display_labels = rf_clf.classes_)
# MLP
mlp_clf = MLPClassifier()
mlp_clf = mlp_clf.fit(X_train, Y_train)
mlp_Y_pred = mlp_clf.predict(X_test)
# print("MLP accuracy: ", accuracy_score(Y_test, mlp_Y_pred))

mlp_cm = confusion_matrix(Y_test, mlp_Y_pred, labels = mlp_clf.classes_)
mlp_disp = ConfusionMatrixDisplay(confusion_matrix = mlp_cm,
                              display_labels = mlp_clf.classes_)

target_names = ['0', '1', '2']

print('Decision Tree:\n', classification_report(Y_test, dtc_Y_pred, target_names=target_names))
print('Decision Tree confusion matrix: \n', confusion_matrix(Y_test, dtc_Y_pred), '\n')
print('SVM:\n', classification_report(Y_test, svm_Y_pred, target_names=target_names))
print('SVM confusion matrix: \n', confusion_matrix(Y_test, svm_Y_pred), '\n')
print('SGD:\n', classification_report(Y_test, sgd_Y_pred, target_names=target_names))
print('SGD confusion matrix: \n', confusion_matrix(Y_test, sgd_Y_pred), '\n')
print('RandomForest:\n', classification_report(Y_test, rf_Y_pred, target_names=target_names))
print('Random Forest confusion matrix: \n', confusion_matrix(Y_test, rf_Y_pred), '\n')
print('MLP:\n', classification_report(Y_test, mlp_Y_pred, target_names=target_names))
print('MLP confusion matrix: \n', confusion_matrix(Y_test, mlp_Y_pred), '\n')

# save the best model (Random Forest)



