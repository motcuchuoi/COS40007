import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 

use_columns = ['Frame', 'Right Hand x', 'Right Hand y', 'Right Hand z', 'Left Hand x', 'Left Hand y', 'Left Hand z']

boning_df = pd.read_csv("Boning.csv")
slicing_df = pd.read_csv("Slicing.csv")

boning_df = boning_df[use_columns]
slicing_df = slicing_df[use_columns]

boning_df['Class'] = 0
slicing_df['Class'] = 1

combined_df = pd.concat([boning_df, slicing_df])
combined_df.to_csv("combined.csv", index = False)

# print(combined_df)

# shuffled_df = combined_df.sample(n=len(combined_df))
# shuffled_df = shuffled_df.reset_index(drop=True)
# print(shuffled_df)

# create composite columns

combined_copy_df = combined_df

# right composite columns
combined_copy_df['R RMS xy'] = np.sqrt((combined_copy_df['Right Hand x']**2 + combined_copy_df['Right Hand y']**2) / 2)
combined_copy_df['R RMS yz'] = np.sqrt((combined_copy_df['Right Hand y']**2 + combined_copy_df['Right Hand z']**2) / 2)
combined_copy_df['R RMS xz'] = np.sqrt((combined_copy_df['Right Hand x']**2 + combined_copy_df['Right Hand z']**2) / 2)
combined_copy_df['R RMS xyz'] = np.sqrt((combined_copy_df['Right Hand x']**2 + combined_copy_df['Right Hand y']**2 + combined_copy_df['Right Hand z']**2) / 3)
combined_copy_df['R Roll'] = (180 * np.arctan2(combined_copy_df['Right Hand y'], np.sqrt(combined_copy_df['Right Hand x']*combined_copy_df['Right Hand x'] + combined_copy_df['Right Hand z']*combined_copy_df['Right Hand z']))/np.pi)
combined_copy_df['R Pitch'] = (180 * np.arctan2(combined_copy_df['Right Hand x'], np.sqrt(combined_copy_df['Right Hand y']*combined_copy_df['Right Hand y'] + combined_copy_df['Right Hand z']*combined_copy_df['Right Hand z']))/np.pi)

# left composite columns
combined_copy_df['L RMS xy'] = np.sqrt((combined_copy_df['Left Hand x']**2 + combined_copy_df['Left Hand y']**2) / 2)
combined_copy_df['L RMS yz'] = np.sqrt((combined_copy_df['Left Hand y']**2 + combined_copy_df['Left Hand z']**2) / 2)
combined_copy_df['L RMS xz'] = np.sqrt((combined_copy_df['Left Hand x']**2 + combined_copy_df['Left Hand z']**2) / 2)
combined_copy_df['L RMS xyz'] = np.sqrt((combined_copy_df['Left Hand x']**2 + combined_copy_df['Left Hand y']**2 + combined_copy_df['Left Hand z']**2) / 3)
combined_copy_df['L Roll'] = (180 * np.arctan2(combined_copy_df['Left Hand y'], np.sqrt(combined_copy_df['Left Hand x']*combined_copy_df['Left Hand x'] + combined_copy_df['Left Hand z'] * combined_copy_df['Left Hand z']))/np.pi)
combined_copy_df['L Pitch'] = (180 * np.arctan2(combined_copy_df['Left Hand x'], np.sqrt(combined_copy_df['Left Hand y']*combined_copy_df['Left Hand y'] + combined_copy_df['Left Hand z']*combined_copy_df['Left Hand z']))/np.pi)

last_column = combined_copy_df.pop('Class')
combined_copy_df['Class'] = last_column

# create new composite file for easy access (for myself...)
# combined_copy_df.to_csv("composite_data.csv", index=False)
# print(combined_copy_df)
