#SCRIPT6 CORRELATION MATRIX FEATURE SELECTION
import pandas as pd
import numpy as np

#url = 'dataset_Final_SMOTE.csv'
url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final_SMOTE.csv'
df = pd.read_csv(url)
print(df.shape)
Y = df['CATEGORY']
print(Y.shape)
del df['CATEGORY']

cor = df.corr().abs()
print(cor.shape)
sum = cor.sum(axis=1)
#print(sum)
index = np.argsort(sum)
print(sum[index])
index_rest = index[0:20]
print(df.columns[index_rest])
#print(sum[index_rest])
colname = df.columns[index_rest]

df_new2 = df[colname]
df_new2['CATEGORY'] = Y
print(df_new2.columns)

#SAVE NEW CSV
df_new2.to_csv('dataset_Final_FS_COR_2.csv', index=False)
