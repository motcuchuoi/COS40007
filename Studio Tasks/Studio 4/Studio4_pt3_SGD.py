import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("combined_data.csv")

feature_cols = df.loc[:, df.columns != 'class']
X = feature_cols
Y = df['class']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size= 0.3, random_state= 1)
# SGD
sgd_clf = SGDClassifier(loss = 'hinge', random_state=1)
sgd_clf.fit(X_train, Y_train)
Y_pred_sgd = sgd_clf.predict(X_test)


SGD_cm = confusion_matrix(Y_test, Y_pred_sgd, labels = sgd_clf.classes_)
SGD_disp = ConfusionMatrixDisplay(confusion_matrix = SGD_cm,
                              display_labels = sgd_clf.classes_)
# print('SGD confusion matrix:\n', confusion_matrix(Y_test, Y_pred_sgd))

# display values of SVM
tp_SGD = np.diag(SGD_cm)

fp_SGD = []
fn_SGD = []
tn_SGD = []

for i in range(len(df['class'].unique())):
    fp_SGD.append(sum(SGD_cm[:,i])- SGD_cm[i,i]) # sum column
    fn_SGD.append(sum(SGD_cm[i,:]) - SGD_cm[i,i]) # sum row

    SGD_temp = np.delete(SGD_cm, i, 0)
    SGD_temp = np.delete(SGD_temp, i, 1)
    tn_SGD.append(sum(sum(SGD_temp)))

print('SGD true positive:\n', tp_SGD, '\n',
      'SGD false positive:\n',  fp_SGD, '\n',
      'SGD false negative:\n', fn_SGD, '\n'
      'SGD true negative:', '\n', tn_SGD
      )


