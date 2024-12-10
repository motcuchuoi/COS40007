# over and undersampling to observe the effect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import classification_report

df = pd.read_csv('combined_data.csv')

# class_weight_dict = {0: 0.2, 1: 0.5, 2: 0.3}

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

clf_smote = RandomForestClassifier(random_state=1)
clf_smote.fit(X1_resampled, Y1_resampled)
Y_pred_smote = clf_smote.predict(X_test)

clf_tomeklink = RandomForestClassifier(random_state=1)
clf_tomeklink.fit(X2_resampled, Y2_resampled)
Y_pred_tomeklink = clf_tomeklink.predict(X_test)

target_names = ['0 - idle', '1 - cutting', '2 - sharpening']
print('rf SMOTE classification report:\n', classification_report(Y_test, Y_pred_smote, target_names=target_names))
print('rf TOMEKLINK classification report:\n', classification_report(Y_test, Y_pred_tomeklink, target_names=target_names))

rf_smote_cm = confusion_matrix(Y_test, Y_pred_smote, labels = clf_smote.classes_)
rf_smote_disp = ConfusionMatrixDisplay(confusion_matrix = rf_smote_cm,
                                  display_labels = clf_smote.classes_)

rf_tomek_cm = confusion_matrix(Y_test, Y_pred_tomeklink, labels = clf_tomeklink.classes_)
rf_tomek_disp = ConfusionMatrixDisplay(confusion_matrix = rf_tomek_cm,
                                  display_labels = clf_tomeklink.classes_)

print('rf with SMOTE confusion matrix: \n', confusion_matrix(Y_test, Y_pred_smote))
print('rf with TomekLinks confusion matrix: \n', confusion_matrix(Y_test, Y_pred_tomeklink))

# display values of rf SMOTE
tp_rf = np.diag(rf_smote_cm)
fp_rf = []
fn_rf = []
tn_rf = []

tp_tomek_rf = np.diag(rf_tomek_cm)
fp_tomek_rf = []
fn_tomek_rf = []
tn_tomek_rf = []

for i in range(len(df['class'].unique())):
    fp_rf.append(sum(rf_smote_cm[:,i])- rf_smote_cm[i,i]) # sum column
    fn_rf.append(sum(rf_smote_cm[i,:]) - rf_smote_cm[i,i]) # sum row
    
    rf_temp = np.delete(rf_smote_cm, i, 0)
    rf_temp = np.delete(rf_temp, i, 1)
    tn_rf.append(sum(sum(rf_temp)))

    fp_tomek_rf.append(sum(rf_tomek_cm[:,i])- rf_tomek_cm[i,i]) # sum column
    fn_tomek_rf.append(sum(rf_tomek_cm[i,:]) - rf_tomek_cm[i,i]) # sum row

    rf_tomek_temp = np.delete(rf_tomek_cm, i, 0)
    rf_tomek_temp = np.delete(rf_tomek_temp, i, 1)
    tn_tomek_rf.append(sum(sum(rf_tomek_temp)))

print('rf (SMOTE) true positive: ', tp_rf, '\n'
      'rf (SMOTE) false positive: ', fp_rf, '\n'
      'rf (SMOTE) false negative: ', fn_rf, '\n'
      'rf (SMOTE) true negative: ', tn_rf, '\n'
      )

print('rf (TOMEKLINKS) true positive: ', tp_tomek_rf, '\n'
      'rf (TOMEKLINKS) false positive: ', fp_tomek_rf, '\n'
      'rf (TOMEKLINKS) false negative: ', fn_tomek_rf, '\n'
      'rf (TOMEKLINKS) true negative: ', tn_tomek_rf, '\n'
      )