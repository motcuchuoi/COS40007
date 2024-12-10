import pickle
import pandas as pd

# load models and dataframe
rf_model = pickle.load(open('vegemite_randomforest_model.pkl', 'rb'))
dtc_model = pickle.load(open('vegemite_decisiontree_model.pkl', 'rb'))
svm_model = pickle.load(open('vegemite_svm_model.pkl', 'rb'))
sgd_model = pickle.load(open('vegemite_sgd_model.pkl', 'rb'))
mlp_model = pickle.load(open('vegemite_mlp_model.pkl', 'rb'))

df = pd.read_csv('vegemite_test_resampled.csv')

# X and Y
X = df.loc[:, df.columns != 'Class']
Y = df['Class']

rf_counter = 0
dtc_counter = 0
svm_counter = 0
sgd_counter = 0
mlp_counter = 0

for i in range(len(X)):
    x_i = X.iloc[i, :].values.reshape(1, -1)
    y_i = Y.iloc[i]

    # Random Forest
    y_pred_rf = rf_model.predict(x_i)[0]
    if y_pred_rf == y_i:
        rf_counter += 1

    # Decision Tree
    y_pred_dtc = dtc_model.predict(x_i)[0]
    if y_pred_dtc == y_i:
        dtc_counter += 1

    # SVM
    y_pred_svm = svm_model.predict(x_i)[0]
    if y_pred_svm == y_i:
        svm_counter += 1

    # SGD
    y_pred_sgd = sgd_model.predict(x_i)[0]
    if y_pred_sgd == y_i:
        sgd_counter += 1

    # MLP
    y_pred_mlp = mlp_model.predict(x_i)[0]
    if y_pred_mlp == y_i:
        mlp_counter += 1

# Calculate accuracies
rf_accuracy = rf_counter / len(Y)
dtc_accuracy = dtc_counter / len(Y)
svm_accuracy = svm_counter / len(Y)
sgd_accuracy = sgd_counter / len(Y)
mlp_accuracy = mlp_counter / len(Y)

print('Random Forest accuracy with 1000 testing data points:', rf_accuracy)
print('Decision Tree accuracy with 1000 testing data points:', dtc_accuracy)
print('SVM accuracy with 1000 testing data points:', svm_accuracy)
print('SGD accuracy with 1000 testing data points:', sgd_accuracy)
print('MLP accuracy with 1000 testing data points:', mlp_accuracy)
