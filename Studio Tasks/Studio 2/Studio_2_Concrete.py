# concrete studio dataset
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import csv 

# concrete_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\concrete.csv")

# # act 1.1
# def concrete_strength(i):
#     if i < 20:
#         return 1
#     elif 20 <= i < 30:
#         return 2
#     elif 30 <= i < 40:
#         return 3
#     elif 40 <= i < 50:
#         return 4
#     elif i >= 50:
#         return 5
#     else:
#         return None
# concrete_df['strength'] = concrete_df['strength'].apply(concrete_strength)

# converted_concrete = concrete_df.copy()
# converted_concrete.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_concrete.csv", index = False)

# cvrt_concrete_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_concrete.csv")

# # act 1.2 
# plt.figure(figsize=(13,6))
# sns.countplot(data = cvrt_concrete_df, x = 'strength', color="b")
# plt.ylabel("Number of samples")
# plt.xlabel("Concrete strength category")
# plt.title("Concrete Strength category distribution")
# # plt.show()

# # act 2.1: categorise age into 12 bins = 12 months (age range is 1-365 (assummingly days))
# age_bins = 12
# cvrt_concrete_df['age'] = cvrt_concrete_df['age'].astype(int)
# cvrt_concrete_df['age'] =  pd.cut (cvrt_concrete_df['age'], bins = age_bins, labels = range(1, age_bins +1))

# # act 2.2: other 7 normalised values
# normalized_columns = [col for col in cvrt_concrete_df if col not in {'age', 'strength'}]

# for col in normalized_columns:
#     cvrt_concrete_df[col] = (cvrt_concrete_df[col] - cvrt_concrete_df[col].min()) / (cvrt_concrete_df[col].max() - cvrt_concrete_df[col].min())
# #print (cvrt_concrete_df)

# normalised_concrete = cvrt_concrete_df.copy()
# normalised_concrete.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\normalised_concrete.csv", index = False)

# #print(normalised_concrete)

# # act 2.3 & 2.4
# norm_con_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\normalised_concrete.csv")

# cement_slag_cov = norm_con_df[['cement', 'slag']].cov().iloc[0, 1]
# cement_ash_cov = norm_con_df[['cement', 'ash']].cov().iloc[0, 1]
# water_fineagg_cov = norm_con_df[['water', 'fineagg']].cov().iloc[0, 1]
# ash_superplastic_cov = norm_con_df[['ash', 'superplastic']].cov().iloc[0, 1]

# norm_con_df['cement_slag'] = cement_slag_cov
# norm_con_df['cement_ash'] = cement_ash_cov
# norm_con_df['water_fineagg'] = water_fineagg_cov
# norm_con_df['ash_superplastic'] = ash_superplastic_cov
# print(norm_con_df)

# norm_con_df.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\features_concrete.csv", index = False)

# # act 3
# feature_concrete_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\features_concrete.csv")
# feature_concrete_df1 = feature_concrete_df.drop(columns = {'slag', 'ash', 'coarseagg', 'fineagg'})

# feature_concrete_df1.to_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\selected_features_concrete.csv", index = False)
# # print(feature_concrete_df1)

# # - - - - - - - - - - - - - - - - - - - - - - - - - #


# act 4.1: converted dataset
converted_columns_name = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
converted_concrete_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_concrete.csv", header=0, names= converted_columns_name)

chosen_converted_columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
X1 = converted_concrete_df[chosen_converted_columns].drop(['strength'], axis = 1)
y1 = converted_concrete_df.strength

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=1) # 70% training and 30% test
clf1 = DecisionTreeClassifier() 
clf1 = clf1.fit(X1_train,y1_train)
y1_pred = clf1.predict(X1_test)

print("Accuracy of Converted dataset:",metrics.accuracy_score(y1_test, y1_pred))

# act 4.2: normalised
normalised_columns_name = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
normalised_concrete_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\normalised_concrete.csv", header=0, names= normalised_columns_name)

chosen_normalised_columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
X2 = normalised_concrete_df[chosen_normalised_columns].drop(['strength'], axis = 1)
y2 = normalised_concrete_df.strength 


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=1) # 70% training and 30% test
clf2 = DecisionTreeClassifier()
clf2 = clf2.fit(X2_train,y2_train)
y2_pred = clf2.predict(X2_test)

print("Accuracy of Normalised dataset:",metrics.accuracy_score(y2_test, y2_pred))

# act 4.3: features dataset
features_columns_name = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength', 'cement_slag', 'cement_ash', 'water_fineagg', 'ash_superplastic']
features_concrete_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\features_concrete.csv", header=0, names= features_columns_name)

chosen_features_columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength', 'cement_slag', 'cement_ash', 'water_fineagg', 'ash_superplastic']
X3 = features_concrete_df[chosen_features_columns].drop(['strength'], axis = 1)
y3 = features_concrete_df.strength


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=1) # 70% training and 30% test
clf3 = DecisionTreeClassifier()
clf3 = clf3.fit(X3_train,y3_train)
y3_pred = clf3.predict(X3_test)

print("Accuracy of Features dataset:",metrics.accuracy_score(y3_test, y3_pred))

# act 4.4: selected features dataset
selected_columns_name = ['cement', 'water', 'superplastic','age', 'strength', 'cement_slag', 'cement_ash', 'water_fineagg', 'ash_superplastic']
selected_concrete_df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\selected_features_concrete.csv", header=0, names= selected_columns_name)

chosen_selected_columns = ['cement', 'water', 'superplastic','age', 'strength', 'cement_slag', 'cement_ash', 'water_fineagg', 'ash_superplastic']
X4 = selected_concrete_df[chosen_selected_columns].drop(['strength'], axis = 1) # Features
y4 = selected_concrete_df.strength 

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.3, random_state=1) # 70% training and 30% test
clf4 = DecisionTreeClassifier()
clf4 = clf4.fit(X4_train,y4_train)
y4_pred = clf4.predict(X4_test)

print("Accuracy of Selected features dataset:",metrics.accuracy_score(y4_test, y4_pred))

# act 4.5
col_names = ['cement' , 'water', 'superplastic', 'age', 'strength']
concrete = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\converted_concrete.csv", header=0, names=col_names)

feature_cols = ['cement', 'water', 'superplastic', 'age', 'strength']
X = concrete [feature_cols].drop(['strength'], axis = 1) # Features
y = concrete.strength # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy of Selected Converted only dataset:",metrics.accuracy_score(y_test, y_pred))
