import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("combined_data.csv")

feature_cols = df.loc[:, df.columns != 'class']
X = feature_cols
Y = df['class']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size= 0.3, random_state= 1)

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

    temp = np.delete(mlp_cm, i, 0)
    temp = np.delete(temp, i, 1)
    tn_mlp.append(sum(sum(temp)))

print('mlp confusion matrix:\n', confusion_matrix(Y_test, Y_pred_mlp))
print('mlp true positive:\n', tp_mlp, '\n',
      'mlp false positive:\n',  fp_mlp, '\n',
      'mlp false negative:\n', fn_mlp, '\n'
      'mlp true negative:', '\n', tn_mlp
      )