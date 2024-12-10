import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest,  f_classif

df = pd.read_csv("vegemite.csv")

class_weight = {0: 300, 1: 350, 2: 350}
# shuffled 
shuffled_df = df.sample(n=len(df))
shuffled_df = shuffled_df.reset_index(drop=True)

test_sample = shuffled_df.groupby('Class').apply(lambda x: x.sample(n = class_weight[x.name])).reset_index(drop=True)
train_model_df = shuffled_df.drop(test_sample.index)

test_sample.to_csv('vegemite_test.csv', index=False)
train_model_df.to_csv("vegemite_train.csv", index = False)
print(train_model_df['Class'].value_counts())

train_df = pd.read_csv('vegemite_train.csv')
train_drop_df = train_df

train_drop_df = train_drop_df.drop(columns= ["TFE Product out temperature", "TFE Steam temperature SP"])
# print out # of unique values
print(train_df.nunique())

# converting unique values into categorical values for columns with few unique values (there should be a faster way to do this surely)
feedlevel_mapping = {25: 0, 45: 1, 50: 2}
motorspeed_mapping = {0: 0, 20: 1, 80: 2}
pump1_mapping = {0: 0, 50: 1, 70: 2, 75: 3, 80: 4}
pump12_mapping = {0: 0, 85: 1, 95: 2, 100: 3}
pump2_mapping = {0: 0, 45: 1, 65: 2, 70: 3, 81: 4}

train_drop_df['FFTE Feed tank level SP'] = train_drop_df['FFTE Feed tank level SP'].map(feedlevel_mapping)
train_drop_df['TFE Motor speed'] = train_drop_df['TFE Motor speed'].map(motorspeed_mapping)
train_drop_df['FFTE Pump 1'] = train_drop_df['FFTE Pump 1'].map(pump1_mapping)
train_drop_df['FFTE Pump 1 - 2'] = train_drop_df['FFTE Pump 1 - 2'].map(pump12_mapping)
train_drop_df['FFTE Pump 2'] = train_drop_df['FFTE Pump 2'].map(pump2_mapping)

print(train_drop_df)
print(train_drop_df.nunique())
print(train_drop_df['Class'].value_counts())
train_drop_df.to_csv('vegemite_converted_train.csv', index = False)
print(train_drop_df)

converted_df = pd.read_csv('vegemite_converted_train.csv')

feature_cols = df.loc[:, converted_df.columns != 'Class']
X = feature_cols
Y = converted_df['Class']

smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X,Y)
resampled_df = pd.concat([pd.DataFrame(X_resampled, columns =X.columns), pd.Series(Y_resampled, name= 'Class')], axis=1)

resampled_df.to_csv("vegemite_resampled.csv", index = 0)
print(resampled_df)
# print(resampled_df['Class'].value_counts())

corr4 = abs(resampled_df.corr()) # correlation matrix
# filtered_corr = corr4.where(corr4 >= 0.5)
lower_triangle4 = np.tril(corr4, k = -1)  # select only the lower triangle of the correlation matrix
mask4 = lower_triangle4 == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (12,10))
sns.heatmap(lower_triangle4, center = 0.5, cmap = 'coolwarm', annot= True, xticklabels = corr4.index, yticklabels = corr4.columns,
            cbar= True, linewidths= 1, mask = mask4)
plt.show()

# mean (solid + density) - can see correlation or the values are relatively close

solid_density_columns = ['FFTE Production solids SP','FFTE Discharge density', 'FFTE Production solids PV',
                        'TFE Production solids PV', 'TFE Production solids SP', 'FFTE Discharge solids', 
                        'TFE Production solids density']

resampled_df['Solid and Density Mean'] = resampled_df[solid_density_columns].mean(axis=1)

# mean pump 1 + pump 2 + pump 12

resampled_df['Pump Mean'] = resampled_df[['FFTE Pump 1', 'FFTE Pump 1 - 2', 'FFTE Pump 2']].mean(axis=1)

# pressure / motor speed
pressure_column = ['FFTE Steam pressure SP', 'TFE Vacuum pressure SP', 'TFE Steam pressure SP',
                   'FFTE Steam pressure PV','TFE Steam pressure PV', 'TFE Vacuum pressure PV']
resampled_df['Pressure Mean'] = resampled_df[pressure_column].mean(axis=1)

pressure_motor_cov = resampled_df[['Pressure Mean', 'TFE Motor speed']].cov().iloc[0, 1]
resampled_df['Pressure and Motor Speed covariance'] = pressure_motor_cov

# temp
temperature_column = ['FFTE Heat temperature 1', 'FFTE Heat temperature 2', 'FFTE Heat temperature 3',
                        'FFTE Temperature 1 - 1', 'FFTE Temperature 1 - 2',
                        'FFTE Temperature 2 - 1', 'FFTE Temperature 2 - 2',
                        'FFTE Temperature 3 - 1', 'FFTE Temperature 3 - 2']

resampled_df['Temperature Mean'] = resampled_df[temperature_column].mean(axis=1)
# resampled_df.to_csv('vegemite_features.csv', index=False)
print(resampled_df.columns)

features_df = pd.read_csv('vegemite_features.csv')

features_df = features_df.astype(float)
feature1_cols = features_df.loc[:, features_df.columns != 'Class']
X1 = feature1_cols
Y1 = features_df['Class']

selector = SelectKBest(f_classif, k=25)
X_feature = selector.fit_transform(X1, Y1)

selected_features = selector.get_support()
selected_features = feature_cols.columns[selected_features]

selected_features_df = features_df.loc[:, selected_features]
selected_features_df['Class'] = features_df['Class']

selected_features_df.to_csv('vegemite_selected_features.csv', index=False)
# print(selected_features_df)



