import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("combined_data.csv")

feature_cols = df.loc[:, df.columns != 'class']
X = feature_cols
Y = df['class']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size= 0.3, random_state= 1)

# randomforest
RF_clf = RandomForestClassifier(random_state=1)
RF_clf.fit(X_train, Y_train)
Y_pred_RF = RF_clf.predict(X_test)

RF_cm = confusion_matrix(Y_test, Y_pred_RF, labels = RF_clf.classes_)
RF_disp = ConfusionMatrixDisplay(confusion_matrix = RF_cm,
                              display_labels= RF_clf.classes_)

# display values of RF
tp_RF = np.diag(RF_cm)

fp_RF = []
fn_RF = []
tn_RF = []

for i in range(len(df['class'].unique())):
    fp_RF.append(sum(RF_cm[:,i])- RF_cm[i,i]) # sum column
    fn_RF.append(sum(RF_cm[i,:]) - RF_cm[i,i]) # sum row

    temp = np.delete(RF_cm, i, 0)
    temp = np.delete(temp, i, 1)
    tn_RF.append(sum(sum(temp)))

print('Random Forest confusion matrix:\n', confusion_matrix(Y_test, Y_pred_RF))
print('RF true positive:\n', tp_RF, '\n',
      'RF false positive:\n',  fp_RF, '\n',
      'RF false negative:\n', fn_RF, '\n'
      'RF true negative:', '\n', tn_RF
      )

