#SCRIPT5 FEATURE SELECTION DECISSION TREE IMPORTANCE 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
import matplotlib.pyplot as plt

#url = 'dataset_Final_SMOTE.csv'
url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final_SMOTE.csv'

df = pd.read_csv(url)
print(df.shape)
Y = df['CATEGORY']
print(Y.shape)
del df['CATEGORY']

X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.3)
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train,y_train)

importance = clf_dt.feature_importances_
index = np.argsort(importance)[-20:]
#print(np.sum(importance[index]))
colname = df.columns[index]
print(colname)
feat_importances = pd.Series(clf_dt.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()

df_new = df[colname]

df_new['CATEGORY'] = Y
print(df_new.shape)

#SAVE NEW CSV
df_new.to_csv('dataset_Final_FS_DT_1.csv', index=False)
