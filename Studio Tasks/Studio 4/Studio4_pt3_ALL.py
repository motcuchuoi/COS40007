import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("combined_data.csv")

feature_cols = df.loc[:, df.columns != 'class']
X = feature_cols
Y = df['class']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size= 0.3, random_state= 1)

# svm
clf = svm.SVC()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

svm_cm = confusion_matrix(Y_test, Y_pred, labels = clf.classes_)
svm_disp = ConfusionMatrixDisplay(confusion_matrix = svm_cm,
                              display_labels = clf.classes_)
# svm_disp.plot()
# plt.show()

# display values of svm
tp_svm = np.diag(svm_cm)
fp_svm = []
fn_svm = []
tn_svm = []

for i in range(len(df['class'].unique())):
    fp_svm.append(sum(svm_cm[:,i])- svm_cm[i,i]) # sum column
    fn_svm.append(sum(svm_cm[i,:]) - svm_cm[i,i]) # sum row

    svm_temp = np.delete(svm_cm, i, 0)
    svm_temp = np.delete(svm_temp, i, 1)
    tn_svm.append(sum(sum(svm_temp)))

print('SVM confusion matrix: \n', confusion_matrix(Y_test, Y_pred))

# sgd
sgd_clf = SGDClassifier(loss = 'hinge', random_state=1)
sgd_clf.fit(X_train, Y_train)
Y_pred_sgd = sgd_clf.predict(X_test)


sgd_cm = confusion_matrix(Y_test, Y_pred_sgd, labels = sgd_clf.classes_)
sgd_disp = ConfusionMatrixDisplay(confusion_matrix = sgd_cm,
                              display_labels = sgd_clf.classes_)

# display values of SVM
tp_sgd = np.diag(sgd_cm)
fp_sgd = []
fn_sgd = []
tn_sgd = []

for i in range(len(df['class'].unique())):
    fp_sgd.append(sum(sgd_cm[:,i])- sgd_cm[i,i]) # sum column
    fn_sgd.append(sum(sgd_cm[i,:]) - sgd_cm[i,i]) # sum row

    sgd_temp = np.delete(sgd_cm, i, 0)
    sgd_temp = np.delete(sgd_temp, i, 1)
    tn_sgd.append(sum(sum(sgd_temp)))

print('SGD confusion matrix: \n', confusion_matrix(Y_test, Y_pred_sgd))

# Randomforest
rf_clf = RandomForestClassifier(random_state=1)
rf_clf.fit(X_train, Y_train)
Y_pred_rf = rf_clf.predict(X_test)

rf_cm = confusion_matrix(Y_test, Y_pred_rf, labels = rf_clf.classes_)
rf_disp = ConfusionMatrixDisplay(confusion_matrix = rf_cm,
                              display_labels= rf_clf.classes_)

# display values of rf
tp_rf = np.diag(rf_cm)
fp_rf = []
fn_rf = []
tn_rf = []

for i in range(len(df['class'].unique())):
    fp_rf.append(sum(rf_cm[:,i])- rf_cm[i,i]) # sum column
    fn_rf.append(sum(rf_cm[i,:]) - rf_cm[i,i]) # sum row

    rf_temp = np.delete(rf_cm, i, 0)
    rf_temp = np.delete(rf_temp, i, 1)
    tn_rf.append(sum(sum(rf_temp)))

print('RandomForest confusion matrix: \n', confusion_matrix(Y_test, Y_pred_rf))

# mlp
mlp_clf = MLPClassifier(random_state=1)
mlp_clf.fit(X_train, Y_train)
Y_pred_mlp = mlp_clf.predict(X_test)

mlp_cm = confusion_matrix(Y_test, Y_pred_mlp, labels = mlp_clf.classes_)
mlp_disp = ConfusionMatrixDisplay(confusion_matrix = mlp_cm,
                              display_labels= mlp_clf.classes_)
# display values of mlp
tp_mlp = np.diag(mlp_cm)
fp_mlp = []
fn_mlp = []
tn_mlp = []

for i in range(len(df['class'].unique())):
    fp_mlp.append(sum(mlp_cm[:,i])- mlp_cm[i,i]) # sum column
    fn_mlp.append(sum(mlp_cm[i,:]) - mlp_cm[i,i]) # sum row

    mlp_temp = np.delete(mlp_cm, i, 0)
    mlp_temp = np.delete(mlp_temp, i, 1)
    tn_mlp.append(sum(sum(mlp_temp)))

print('MLP confusion matrix: \n', confusion_matrix(Y_test, Y_pred_mlp), '\n')

# all positive and negative values
print('SVM true positive: ', tp_svm, '\n'
      'SVM false positive: ', fp_svm, '\n'
      'SVM false negative: ', fn_svm, '\n'
      'SVM true negative: ', tn_svm, '\n'
      )

print('SGD true positive: ', tp_sgd, '\n'
      'SGD false positive: ', fp_sgd, '\n'
      'SGD false negative: ', fn_sgd, '\n'
      'SGD true negative: ', tn_sgd, '\n'
      )

print('RF true positive: ', tp_rf, '\n'
      'RF false positive: ', fp_rf, '\n'
      'RF false negative: ', fn_rf, '\n'
      'RF true negative: ', tn_rf, '\n'
      )

print('MLP true positive: ', tp_mlp, '\n'
      'MLP false positive: ', fp_mlp, '\n'
      'MLP false negative: ', fn_mlp, '\n'
      'MLP true negative: ', tn_mlp, '\n'
      )