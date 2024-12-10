import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# read CSV file
df = pd.read_csv(r"C:\Users\Dell\OneDrive - Swinburne University\Desktop\COS40007\water_potability.csv")

# check duplicate
df.duplicated().sum()

# View the duplicate records
duplicates = df.duplicated()
df[duplicates]

# check outliers
#df.boxplot(column= ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'], rot = 45, figsize= (20,10))

# defined outliers
df_outliers = pd.DataFrame(df.loc[:,])

# Calculate IQR
Q1 = df_outliers.quantile(0.25)
Q3 = df_outliers.quantile(0.75)
IQR = Q3 - Q1

df.columns

#Replace every outlier on the upper side by the upper whisker - for 'water', 'superplastic', 
# 'fineagg', 'age' and 'strength' columns
for i, j in zip(np.where(df_outliers > Q3 + 1.5 * IQR)[0], np.where(df_outliers > Q3 + 1.5 * IQR)[1]):
    
    whisker  = Q3 + 1.5 * IQR
    df_outliers.iloc[i,j] = whisker[j]
    
# Replace every outlier on the lower side by the lower whisker - for 'water' column
for i, j in zip(np.where(df_outliers < Q1 - 1.5 * IQR)[0], np.where(df_outliers < Q1 - 1.5 * IQR)[1]): 
    
    whisker  = Q1 - 1.5 * IQR
    df_outliers.iloc[i,j] = whisker[j]

df.drop(columns = df.loc[:,], inplace = True)
df = pd.concat([df, df_outliers], axis = 1)

df.boxplot(column= ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'], rot = 45, figsize= (20,10))

# check null
df.isnull().sum()

# replace null with 0
df.fillna(0).isnull().sum()

# print(df.duplicated().sum(), df[duplicates])
# plt.show()
# print(IQR)
# print(df.columns)

# print(df.fillna(0).isnull().sum())

# predictors: pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, TurbidityPotabilty

cols = [i for i in df.columns if i not in 'Potability']
length = len(cols)
cs = ["b","r","g","c","m","k","lime","c"]
fig = plt.figure(figsize=(13,25))

for i,j,k in itertools.zip_longest(cols,range(length),cs):
    plt.subplot(3,3,j+1)
    ax = sns.distplot(df[i],color=k,rug=True)
    ax.set_facecolor("w")
    plt.axvline(df[i].mean(),linestyle="dashed",label="mean",color="k")
    plt.legend(loc="best")
    plt.title(i,color="navy")
    plt.xlabel("")

for x in df:
    sns.distplot(df[x])
    plt.title("{} distribution".format(x))

# plt.figure(figsize=(13,6))
# sns.distplot(df["Potability"],color="b",rug=True)
# plt.axvline(df["Potability"].mean(), linestyle="dashed",color="k", label='mean',linewidth=2)
# plt.legend(loc="best",prop={"size":14})
# plt.title("Water Potability distribution")
# plt.show()

# # statistic table
# df.describe().T
# print(df.describe())

sns.pairplot(df, diag_kind = 'kde', corner = True, plot_kws ={"s": 1})
plt.show()


