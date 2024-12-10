import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

df = pd.read_csv('vegemite_selected_features.csv')

sp_features = [col for col in df.columns if col.endswith('SP')]
sp_df = df[sp_features]

X_train, X_test, y_train, y_test = train_test_split(sp_df, df['Class'], test_size=0.3, random_state=42)

sp_tree_model = DecisionTreeClassifier(random_state=42)
sp_tree_model.fit(X_train, y_train)

tree_rules = export_text(sp_tree_model, feature_names=sp_features)
print(tree_rules)
