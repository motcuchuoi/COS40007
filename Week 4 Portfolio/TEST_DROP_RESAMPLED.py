import pandas as pd

df = pd.read_csv('vegemite_test.csv')

df = df.drop(columns= ["TFE Product out temperature", "TFE Steam temperature SP"])
# print out # of unique values

feedlevel_mapping = {25: 0, 45: 1, 50: 2}
motorspeed_mapping = {0: 0, 20: 1, 80: 2}
pump1_mapping = {0: 0, 50: 1, 70: 2, 75: 3, 80: 4}
pump12_mapping = {0: 0, 85: 1, 95: 2, 100: 3}
pump2_mapping = {0: 0, 45: 1, 65: 2, 70: 3, 81: 4}

df['FFTE Feed tank level SP'] = df['FFTE Feed tank level SP'].map(feedlevel_mapping)
df['TFE Motor speed'] = df['TFE Motor speed'].map(motorspeed_mapping)
df['FFTE Pump 1'] = df['FFTE Pump 1'].map(pump1_mapping)
df['FFTE Pump 1 - 2'] = df['FFTE Pump 1 - 2'].map(pump12_mapping)
df['FFTE Pump 2'] = df['FFTE Pump 2'].map(pump2_mapping)

# df.to_csv('vegemite_test_resampled.csv', index=False)
df_train = pd.read_csv('vegemite_resampled.csv')

# Compare the columns of the two DataFrames
columns_resampled = set(df.columns)
columns_train = set(df_train.columns)

# Find the difference in columns between the two DataFrames
diff_columns = columns_resampled.symmetric_difference(columns_train)

# Print the difference in columns
print("Columns that differ between the two files:", diff_columns)