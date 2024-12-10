# over and undersampling to observe the effect
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

df = pd.read_csv('combined_data.csv')

feature_cols = df.loc[:, df.columns != 'class']
X = feature_cols
Y = df['class']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size= 0.3, random_state= 1)

# oversampling minority classes using SMOTE
smote = SMOTE(random_state=42)
X1_resampled, Y1_resampled = smote.fit_resample(X, Y)

# undersampling majority classes using Tomeklinks
tl = TomekLinks(sampling_strategy='majority')
X2_resampled, Y2_resampled = tl.fit_resample(X, Y)

clf_smote = svm.SVC()
clf_smote.fit(X1_resampled, Y1_resampled)
Y_pred_smote = clf_smote.predict(X_test)

clf_tomeklink = svm.SVC()
clf_tomeklink.fit(X2_resampled, Y2_resampled)
Y_pred_tomeklink = clf_tomeklink.predict(X_test)

svm_smote_cm = confusion_matrix(Y_test, Y_pred_smote, labels = clf_smote.classes_)
svm_smote_disp = ConfusionMatrixDisplay(confusion_matrix = svm_smote_cm,
                                  display_labels = clf_smote.classes_)

svm_tomek_cm = confusion_matrix(Y_test, Y_pred_tomeklink, labels = clf_tomeklink.classes_)
svm_tomek_disp = ConfusionMatrixDisplay(confusion_matrix = svm_tomek_cm,
                                  display_labels = clf_tomeklink.classes_)

print('SVM with SMOTE confusion matrix: \n', confusion_matrix(Y_test, Y_pred_smote))
print('SVM with TomekLinks confusion matrix: \n', confusion_matrix(Y_test, Y_pred_tomeklink))

# display values of svm SMOTE
tp_svm = np.diag(svm_smote_cm)
fp_svm = []
fn_svm = []
tn_svm = []

tp_tomek_svm = np.diag(svm_tomek_cm)
fp_tomek_svm = []
fn_tomek_svm = []
tn_tomek_svm = []

for i in range(len(df['class'].unique())):
    fp_svm.append(sum(svm_smote_cm[:,i])- svm_smote_cm[i,i]) # sum column
    fn_svm.append(sum(svm_smote_cm[i,:]) - svm_smote_cm[i,i]) # sum row
    
    svm_temp = np.delete(svm_smote_cm, i, 0)
    svm_temp = np.delete(svm_temp, i, 1)
    tn_svm.append(sum(sum(svm_temp)))

    fp_tomek_svm.append(sum(svm_tomek_cm[:,i])- svm_tomek_cm[i,i]) # sum column
    fn_tomek_svm.append(sum(svm_tomek_cm[i,:]) - svm_tomek_cm[i,i]) # sum row

    svm_tomek_temp = np.delete(svm_tomek_cm, i, 0)
    svm_tomek_temp = np.delete(svm_tomek_temp, i, 1)
    tn_tomek_svm.append(sum(sum(svm_tomek_temp)))

print('SVM (SMOTE) true positive: ', tp_svm, '\n'
      'SVM (SMOTE) false positive: ', fp_svm, '\n'
      'SVM (SMOTE) false negative: ', fn_svm, '\n'
      'SVM (SMOTE) true negative: ', tn_svm, '\n'
      )

print('SVM (TOMEKLINKS) true positive: ', tp_tomek_svm, '\n'
      'SVM (TOMEKLINKS) false positive: ', fp_tomek_svm, '\n'
      'SVM (TOMEKLINKS) false negative: ', fn_tomek_svm, '\n'
      'SVM (TOMEKLINKS) true negative: ', tn_tomek_svm, '\n'
      )
# # svm
# clf = svm.SVC()
# clf.fit(X_train, Y_train)
# Y_pred = clf.predict(X_test)

# svm_cm = confusion_matrix(Y_test, Y_pred, labels = clf.classes_)
# svm_disp = ConfusionMatrixDisplay(confusion_matrix = svm_cm,
#                               display_labels = clf.classes_)
# # svm_disp.plot()
# # plt.show()

# # display values of svm
# tp_svm = np.diag(svm_cm)
# fp_svm = []
# fn_svm = []
# tn_svm = []

# for i in range(len(df['class'].unique())):
#     fp_svm.append(sum(svm_cm[:,i])- svm_cm[i,i]) # sum column
#     fn_svm.append(sum(svm_cm[i,:]) - svm_cm[i,i]) # sum row

#     svm_temp = np.delete(svm_cm, i, 0)
#     svm_temp = np.delete(svm_temp, i, 1)
#     tn_svm.append(sum(sum(svm_temp)))

# print('SVM confusion matrix: \n', confusion_matrix(Y_test, Y_pred))



