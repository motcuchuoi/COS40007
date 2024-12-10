import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import simpson

df = pd.read_csv("composite_data.csv")

# create mean with 60 frame = 1 mean value
columns_list = []

frames_per_minute = 60

for i in range(0, len(df), frames_per_minute):
    # dividing into segment, making sure it runs 60 frames
    df_segment = df.iloc[i:i+frames_per_minute]
    
    feature_dict = {}

    feature_dict['Class'] = df_segment['Class'].iloc[0]

    for column in df.columns:
        if column != 'Frame' and column != 'Class':
            data = df_segment[column].values

            feature_dict[f"{column} mean"] = np.mean(data)
            feature_dict[f"{column} std"] = np.std(data)
            feature_dict[f"{column} max"] = np.max(data)
            feature_dict[f"{column} min"] = np.min(data)
            feature_dict[f"{column} AUC"] = simpson(data)
            feature_dict[f"{column} peaks"] = len(find_peaks(data)[0])

    columns_list.append(feature_dict)

df1 = pd.DataFrame(columns_list)
df1.to_csv("statistical_data.csv", index=False)

print(df1)
