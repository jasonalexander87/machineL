#SCRIPT4 CREATE NEW DATASET WITH SMOTE
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE 

url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final_FE.csv'
#url = 'dataset_Final_FE.csv'
df = pd.read_csv(url)

Y = df.CATEGORY
del df['CATEGORY']
del df['class']
del df['num_outbound_cmds']
X = df

smi = SMOTE(random_state=42)
X_res, Y_res = smi.fit_resample(X, Y)
print(X_res.shape)
print(Y_res.shape)
X_res['CATEGORY'] = Y_res
print(X_res.shape)

X_res.to_csv('dataset_Final_SMOTE.csv', index=False)