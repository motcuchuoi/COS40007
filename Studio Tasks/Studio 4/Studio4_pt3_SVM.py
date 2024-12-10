import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("combined_data.csv")

feature_cols = df.loc[:, df.columns != 'class']
X = feature_cols
Y = df['class']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size= 0.3, random_state= 1)

# SVM
clf = svm.SVC()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

# print('SVM confusion matrix:\n', confusion_matrix(Y_test, Y_pred))

SVM_cm = confusion_matrix(Y_test, Y_pred, labels = clf.classes_)
SVM_disp = ConfusionMatrixDisplay(confusion_matrix = SVM_cm,
                              display_labels = clf.classes_)
# SVM_disp.plot()
# plt.show()

# display values of SVM
tp_SVM = np.diag(SVM_cm)

fp_SVM = []
fn_SVM = []
tn_SVM = []

for i in range(len(df['class'].unique())):
    fp_SVM.append(sum(SVM_cm[:,i])- SVM_cm[i,i]) # sum column
    fn_SVM.append(sum(SVM_cm[i,:]) - SVM_cm[i,i]) # sum row

    temp = np.delete(SVM_cm, i, 0)
    temp = np.delete(temp, i, 1)
    tn_SVM.append(sum(sum(temp)))

print('SVM true positive:\n', tp_SVM, '\n',
      'SVM false positive:\n',  fp_SVM, '\n',
      'SVM false negative:\n', fn_SVM, '\n'
      'SVM true negative:', '\n', tn_SVM
      )
