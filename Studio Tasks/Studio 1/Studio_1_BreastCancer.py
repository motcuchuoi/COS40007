import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import csv 

# read CSV file
df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\databreast.csv")
# df_headings = ["ID", "Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1", "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2', "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3" ]

# with open(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\databreast.csv", mode='r', newline='') as file:
#     reader = csv.reader(file)
#     data = list(reader)

# # Write the new data with the column headings
# with open(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\databreast.csv", mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(df_headings)  # Write the column headings
#     writer.writerows(data)  # Write the original data

# # check duplicate
# df.duplicated().sum()
# print(df.duplicated().sum())

# View the duplicate records
#duplicates = df.duplicated()
#df[duplicates]
#print(duplicates)

df_copy = df
df1 = df_copy.drop(columns= "ID")

mapping = {'M': 1, 'B': 0}
df1["Diagnosis"] = df1["Diagnosis"].map(mapping)
#df1.boxplot(column= ["Diagnosis", "radius1", "texture1", "perimeter1", "area1", 'smoothness1', "compactness1", "concavity1", "concave_point1", "symmetry1", "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2", 'smoothness2', "compactness2", "concavity2", "concave_point2", "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", "concavity3", "concave_point3", "symmetry3", "fractal_dimension3" ], rot = 45, figsize= (20,10))

# defined outliers
df1_outliers = pd.DataFrame(df1.loc[:,])

# Calculate IQR
Q1 = df1_outliers.quantile(0.25)
Q3 = df1_outliers.quantile(0.75)
IQR = Q3 - Q1

# print(IQR)

# df1.columns
# print(df1.columns)

#Replace every outlier on the upper side by the upper whisker - for 'water', 'superplastic', 
# 'fineagg', 'age' and 'strength' columns
for i, j in zip(np.where(df1_outliers > Q3 + 1.5 * IQR)[0], np.where(df1_outliers > Q3 + 1.5 * IQR)[1]):
    
    whisker  = Q3 + 1.5 * IQR
    df1_outliers.iloc[i,j] = whisker[j]
    
# Replace every outlier on the lower side by the lower whisker - for 'water' column
for i, j in zip(np.where(df1_outliers < Q1 - 1.5 * IQR)[0], np.where(df1_outliers < Q1 - 1.5 * IQR)[1]): 
    
    whisker  = Q1 - 1.5 * IQR
    df1_outliers.iloc[i,j] = whisker[j]

df1.drop(columns = df1.loc[:,], inplace = True)
df1 = pd.concat([df1, df1_outliers], axis = 1)

# df1.boxplot(column= ['Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1',
#        'smoothness1', 'compactness1', 'concavity1', 'concave_point1',
#        'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2',
#        'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_point2',
#        'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
#        'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_point3',
#        'symmetry3', 'fractal_dimension3'], rot = 45, figsize= (20,10))

# check null
# df.isnull().sum()

# # replace null with 0
# df.fillna(0).isnull().sum()

# # print(df.duplicated().sum(), df[duplicates])
# # plt.show()
# # print(IQR)
# # print(df.columns)

# # print(df.fillna(0).isnull().sum())

# cols = [i for i in df1.columns if i not in 'Diagnosis']
# length = len(cols)
# cs = ["b","r","g","c","m","k","lime","c"]
# fig = plt.figure(figsize=(13,25))

# for i,j,k in itertools.zip_longest(cols,range(length),cs):
#     plt.subplot(4,8,j+1)
#     ax = sns.distplot(df1[i],color=k,rug=True)
#     ax.set_facecolor("w")
#     plt.axvline(df1[i].mean(),linestyle="dashed",label="mean",color="k")
#     plt.legend(loc="best")
#     plt.title(i,color="navy")
#     plt.xlabel("")

# for x in df1:
#     sns.distplot(df1[x])
#     plt.title("{} distribution".format(x))

# plt.figure(figsize=(13,6))
# sns.distplot(df1["Diagnosis"],color="b",rug=True)
# plt.axvline(df1["Diagnosis"].mean(), linestyle="dashed",color="k", label='mean',linewidth=2)
# plt.legend(loc="best",prop={"size":14})
# plt.title("Breast Cancer diagnosis distribution")
# plt.show()

# # # statistic table
# # df.describe().T
# # print(df.describe())

# pair plot generation
# sns.pairplot(df1, diag_kind = 'kde', corner = True, plot_kws ={"s": 0.5})
# plt.show()

# df1.corr()
# print(df1.corr)

# df1.describe().T
# print(df1.describe().T)

# sns.pairplot(df1[['Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1',
#        'smoothness1', 'compactness1', 'concavity1', 'concave_point1',
#        'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2',
#        'area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_point2',
#        'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
#        'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_point3',
#        'symmetry3', 'fractal_dimension3']], kind = 'reg', corner = True)
# plt.show()
# df1_copy = df1
# df2 = df1_copy(columns= ['Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_point1', 'symmetry1', 'fractal_dimension1'])

df1_copy = df1

df2 = df1_copy[['Diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_point1','symmetry1', 'fractal_dimension1']].copy()
df3 = df1_copy[['radius2', 'texture2', 'perimeter2','area2', 'smoothness2', 'compactness2', 'concavity2', 'concave_point2','symmetry2', 'fractal_dimension2']].copy()
df4 = df1_copy[['radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_point3', 'symmetry3', 'fractal_dimension3']].copy()

corr2 = abs(df2.corr()) # correlation matrix
lower_triangle2 = np.tril(corr2, k = -1)  # select only the lower triangle of the correlation matrix
mask2 = lower_triangle2 == 0  # to mask the upper triangle in the following heatmap


plt.figure(figsize = (12,10))
sns.heatmap(lower_triangle2, center = 0.5, cmap = 'coolwarm', annot= True, xticklabels = corr2.index, yticklabels = corr2.columns,
            cbar= True, linewidths= 1, mask2 = mask2)   # Da Heatmap
plt.show()