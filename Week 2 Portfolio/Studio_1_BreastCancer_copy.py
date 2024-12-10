import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import itertools
import csv 

# read CSV file
df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\databreast.csv")

df_copy = df
df1 = df_copy.drop(columns= "ID")

mapping = {'M': 1, 'B': 0}
df1["Diagnosis"] = df1["Diagnosis"].map(mapping)
#df1.boxplot(column= ["Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1", "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2', "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3" ], rot = 45, figsize= (20,10))

corr = abs(df1.corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (12,10))
sns.heatmap(lower_triangle, center = 0.5, cmap = 'coolwarm', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= True, linewidths= 1, mask = mask)   # Da Heatmap
# plt.show()

converted_databreast = df1.copy()
converted_databreast.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_databreast.csv", index = False)

cdb_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_databreast.csv")

cdb_df_copy = cdb_df.copy()
consel_df = cdb_df_copy[['Diagnosis', 'radius1', 'radius2', 'radius3', 'perimeter1','perimeter2', 'perimeter3',
                            'compactness1', 'compactness2', 'compactness3', 'concavity1', 'concavity2', 'concavity3', 
                            'concave_point1', 'concave_point2', 'concave_point3', 'area1', 'area2', 'area3']].copy()

consel_df.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_selected_databreast.csv", index = False)
# plt.figure(figsize=(13,6))
# sns.countplot(data = cdb_df, x = 'Diagnosis', color="b")
# plt.ylabel("Number of samples")
# plt.xlabel("Breast Cancer Diagnosis")
# plt.title("Breast Cancer Diagnosis distribution")
# # plt.show()

# normalised dataset
normalised_columns = [col for col in cdb_df if col not in 'Diagnosis']

for col in normalised_columns:
    cdb_df[col] = (cdb_df[col] - cdb_df[col].min()) / (cdb_df[col].max() - cdb_df[col].min())
# print (cdb_df)

cdb_df.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\normalised_databreast.csv", index = False)
print(cdb_df)
# feature dataset

norm_db_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\normalised_databreast.csv")

# categorise + covariance value calculated
db_radius_cov = norm_db_df[['radius1', 'radius2', 'radius3']].cov().iloc[0, 1]
db_texture_cov = norm_db_df[['texture1', 'texture2', 'texture3']].cov().iloc[0, 1]
db_perimeter_cov = norm_db_df[['perimeter1','perimeter2', 'perimeter3']].cov().iloc[0,1]
db_area_cov = norm_db_df[['area1', 'area2', 'area3']].cov().iloc[0, 1]
db_smoothness_cov = norm_db_df[['smoothness1', 'smoothness2', 'smoothness3']].cov().iloc[0, 1]
db_compactness_cov = norm_db_df[['compactness1', 'compactness2', 'compactness3']].cov().iloc[0, 1]
db_concavity_cov = norm_db_df[['concavity1', 'concavity2', 'concavity3']].cov().iloc[0, 1]
db_concavept_cov = norm_db_df[['concave_point1', 'concave_point2', 'concave_point3']].cov().iloc[0, 1]
db_symmetry_cov = norm_db_df[['symmetry1', 'symmetry2', 'symmetry3']].cov().iloc[0, 1]
db_fracdim_cov = norm_db_df[['fractal_dimension1', 'fractal_dimension2', 'fractal_dimension3']].cov().iloc[0, 1]

# assign values
norm_db_df['radius_sum'] = db_radius_cov
norm_db_df['texture_sum'] = db_texture_cov
norm_db_df['perimeter_sum'] = db_perimeter_cov
norm_db_df['area_sum'] = db_area_cov
norm_db_df['smoothness_sum'] = db_smoothness_cov
norm_db_df['compactness_sum'] = db_compactness_cov
norm_db_df['concavity_sum'] = db_concavity_cov
norm_db_df['concave_point_sum'] = db_concavept_cov
norm_db_df['symmetry_sum'] = db_symmetry_cov
norm_db_df['fractual_dimension_sum'] = db_fracdim_cov

# print(norm_db_df)

# save new dataset
norm_db_df.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\features_databreast.csv", index = False)

selected_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\features_databreast.csv")
selected_df2 = selected_df[['Diagnosis', 'radius1', 'radius2', 'radius3', 'perimeter1','perimeter2', 'perimeter3',
                            'compactness1', 'compactness2', 'compactness3', 'concavity1', 'concavity2', 'concavity3', 
                            'concave_point1', 'concave_point2', 'concave_point3', 'area1', 'area2', 'area3', 
                            'radius_sum', 'texture_sum', 'perimeter_sum', 'area_sum', 'smoothness_sum', 'compactness_sum', 
                            'concavity_sum', 'concave_point_sum', 'symmetry_sum', 'fractual_dimension_sum']].copy()

selected_df2.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\selected_features_databreast.csv", index = False)

# Converted dataset
converted_columns_name = ["Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1",
                        "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2',
                        "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3",
                        "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3"]

converted_databreast_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_databreast.csv", header=0, names= converted_columns_name)

chosen_converted_columns = ["Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1",
                        "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2',
                        "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3",
                        "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3"]
X1 = converted_databreast_df[chosen_converted_columns].drop(['Diagnosis'], axis = 1)
y1 = converted_databreast_df.Diagnosis

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=1) # 70% training and 30% test
clf1 = DecisionTreeClassifier() 
clf1 = clf1.fit(X1_train,y1_train)
y1_pred = clf1.predict(X1_test)

print("Accuracy of Converted Breast Cancer dataset:",metrics.accuracy_score(y1_test, y1_pred))

# Normalised dataset
normalised_columns_name = ["Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1",
                        "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2',
                        "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3",
                        "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3"]

normalised_databreast_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\normalised_databreast.csv", header=0, names= normalised_columns_name)

chosen_normalised_columns = ["Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1",
                        "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2',
                        "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3",
                        "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3"]
X2 = normalised_databreast_df[chosen_normalised_columns].drop(['Diagnosis'], axis = 1)
y2 = normalised_databreast_df.Diagnosis

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=1) # 70% training and 30% test
clf2 = DecisionTreeClassifier() 
clf2 = clf2.fit(X2_train,y2_train)
y2_pred = clf2.predict(X2_test)

print("Accuracy of Normalised Breast Cancer dataset:",metrics.accuracy_score(y2_test, y2_pred))

# Features dataset
features_columns_name = ["Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1",
                        "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2',
                        "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3",
                        "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3",
                        'radius_sum', 'texture_sum', 'perimeter_sum', 'area_sum', 'smoothness_sum', 'compactness_sum', 
                        'concavity_sum', 'concave_point_sum', 'symmetry_sum', 'fractual_dimension_sum']
features_databreast_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\features_databreast.csv", header=0, names= features_columns_name)

chosen_features_columns = ["Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1",
                        "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2',
                        "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3",
                        "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3",
                        'radius_sum', 'texture_sum', 'perimeter_sum', 'area_sum', 'smoothness_sum', 'compactness_sum', 
                        'concavity_sum', 'concave_point_sum', 'symmetry_sum', 'fractual_dimension_sum']
X3 = features_databreast_df[chosen_features_columns].drop(['Diagnosis'], axis = 1)
y3 = features_databreast_df.Diagnosis

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=1) # 70% training and 30% test
clf3 = DecisionTreeClassifier() 
clf3 = clf3.fit(X3_train,y3_train)
y3_pred = clf3.predict(X3_test)

print("Accuracy of Features Breast Cancer dataset:",metrics.accuracy_score(y3_test, y3_pred))

# Selected Feature dataset
selected_columns_name = ['Diagnosis', 'radius1', 'radius2', 'radius3', 'perimeter1','perimeter2', 'perimeter3',
                            'compactness1', 'compactness2', 'compactness3', 'concavity1', 'concavity2', 'concavity3', 
                            'concave_point1', 'concave_point2', 'concave_point3', 'area1', 'area2', 'area3', 
                            'radius_sum', 'texture_sum', 'perimeter_sum', 'area_sum', 'smoothness_sum', 'compactness_sum', 
                            'concavity_sum', 'concave_point_sum', 'symmetry_sum', 'fractual_dimension_sum']
selected_databreast_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\selected_features_databreast.csv", header=0, names= selected_columns_name)

chosen_selected_columns = ['Diagnosis', 'radius1', 'radius2', 'radius3', 'perimeter1','perimeter2', 'perimeter3',
                            'compactness1', 'compactness2', 'compactness3', 'concavity1', 'concavity2', 'concavity3', 
                            'concave_point1', 'concave_point2', 'concave_point3', 'area1', 'area2', 'area3', 
                            'radius_sum', 'texture_sum', 'perimeter_sum', 'area_sum', 'smoothness_sum', 'compactness_sum', 
                            'concavity_sum', 'concave_point_sum', 'symmetry_sum', 'fractual_dimension_sum']
X4 = selected_databreast_df[chosen_selected_columns].drop(['Diagnosis'], axis = 1)
y4 = selected_databreast_df.Diagnosis

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=1) # 70% training and 30% test
clf4 = DecisionTreeClassifier() 
clf4 = clf4.fit(X4_train,y4_train)
y4_pred = clf4.predict(X4_test)

print("Accuracy of Selected Features Breast Cancer dataset:",metrics.accuracy_score(y4_test, y4_pred))

# Converted Selected dataset
consel_columns_name = ['Diagnosis', 'radius1', 'radius2', 'radius3', 'perimeter1','perimeter2', 'perimeter3',
                            'compactness1', 'compactness2', 'compactness3', 'concavity1', 'concavity2', 'concavity3', 
                            'concave_point1', 'concave_point2', 'concave_point3', 'area1', 'area2', 'area3']

consel_databreast_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_selected_databreast.csv", header=0, names= consel_columns_name)

consel_columns = ['Diagnosis', 'radius1', 'radius2', 'radius3', 'perimeter1','perimeter2', 'perimeter3',
                            'compactness1', 'compactness2', 'compactness3', 'concavity1', 'concavity2', 'concavity3', 
                            'concave_point1', 'concave_point2', 'concave_point3', 'area1', 'area2', 'area3']

X = consel_databreast_df[consel_columns].drop(['Diagnosis'], axis = 1)
y = consel_databreast_df.Diagnosis

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
clf = DecisionTreeClassifier() 
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy of Selected Converted Breast Cancer dataset:",metrics.accuracy_score(y_test, y_pred))

# print(selected_df2)

# # Calculate IQR
# Q1 = df1_outliers.quantile(0.25)
# Q3 = df1_outliers.quantile(0.75)
# IQR = Q3 - Q1

# # print(IQR)

# for i, j in zip(np.where(df1_outliers > Q3 + 1.5 * IQR)[0], np.where(df1_outliers > Q3 + 1.5 * IQR)[1]):
    
#     whisker  = Q3 + 1.5 * IQR
#     df1_outliers.iloc[i,j] = whisker[j]
    
# # Replace every outlier on the lower side by the lower whisker - for 'water' column
# for i, j in zip(np.where(df1_outliers < Q1 - 1.5 * IQR)[0], np.where(df1_outliers < Q1 - 1.5 * IQR)[1]): 
    
#     whisker  = Q1 - 1.5 * IQR
#     df1_outliers.iloc[i,j] = whisker[j]

# df1.drop(columns = df1.loc[:,], inplace = True)
# df1 = pd.concat([df1, df1_outliers], axis = 1)

# df1_copy = df1

# df2 = df1_copy[['Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_point1','symmetry1', 'fractal_dimension1']].copy()
# df3 = df1_copy[['Diagnosis', 'radius2', 'texture2', 'perimeter2','area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_point2','symmetry2', 'fractal_dimension2']].copy()
# df4 = df1_copy[['Diagnosis', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_point3', 'symmetry3', 'fractal_dimension3']].copy()

# # 2
# corr2 = abs(df2.corr()) # correlation matrix
# lower_triangle2 = np.tril(corr2, k = -1)  # select only the lower triangle of the correlation matrix
# mask2 = lower_triangle2 == 0  # to mask the upper triangle in the following heatmap

# plt.figure(figsize = (12,10))
# sns.heatmap(lower_triangle2, center = 0.5, cmap = 'coolwarm', annot= True, xticklabels = corr2.index, yticklabels = corr2.columns,
#             cbar= True, linewidths= 1, mask = mask2)   # Da Heatmap

# # 3
# corr3 = abs(df3.corr()) # correlation matrix
# lower_triangle3 = np.tril(corr3, k = -1)  # select only the lower triangle of the correlation matrix
# mask3 = lower_triangle3 == 0  # to mask the upper triangle in the following heatmap

# plt.figure(figsize = (12,10))
# sns.heatmap(lower_triangle3, center = 0.5, cmap = 'coolwarm', annot= True, xticklabels = corr3.index, yticklabels = corr3.columns,
#             cbar= True, linewidths= 1, mask = mask3)   # Da Heatmap

# # 4
# corr4 = abs(df4.corr()) # correlation matrix
# lower_triangle4 = np.tril(corr4, k = -1)  # select only the lower triangle of the correlation matrix
# mask4 = lower_triangle4 == 0  # to mask the upper triangle in the following heatmap


# plt.figure(figsize = (12,10))
# sns.heatmap(lower_triangle4, center = 0.5, cmap = 'coolwarm', annot= True, xticklabels = corr4.index, yticklabels = corr4.columns,
#             cbar= True, linewidths= 1, mask = mask4)   # Da Heatmap

# # plt.show()
