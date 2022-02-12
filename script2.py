#SCRIPT2 FEATURE ENGINIRING
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher



#FUNCTION TO NORMALIZE
def normalize(collumns,df_test):
  for i in collumns:
    df_test.iloc[:,i]= (df_test.iloc[:,i] - df_test.iloc[:,i].min()) / (df_test.iloc[:,i].max() - df_test.iloc[:,i].min())    



url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final.csv'
#url = 'dataset_Final.csv'
df = pd.read_csv(url)

#ALL ZEROOES NO INFO SO DROP
#print(df['num_outbound_cmds'])
#COLUMNS TO NORMALIZE
to_normalize = [4,5,7,8,9,10,12,15,16,17,18,19,22,23,31,32]

#NORMALIZATION ON COLUMNS
normalize(to_normalize,df)

#HASH TRICK 5 NEW FEATURES
h = FeatureHasher(n_features=5, input_type='string')
f = h.transform(df.service)

df_hash = pd.DataFrame(f.toarray(),columns=['HASH1', 'HASH2','HASH3','HASH4','HASH5'])

df = df.join(df_hash)

#ONE HOT FOR PROTOCOLTYPE
one_hot = pd.get_dummies(df['protocol_type'])
df = df.join(one_hot)

#BUIISSNES LOGIC ON FLAG
df.loc[df["flag"] != "SF",'flag'] = 0
df.loc[df["flag"] == "SF", 'flag'] = 1


#DROP COLUMNS NOT NEEDED protocol_type , service
del df['service']
del df['protocol_type']

#SAVE NEW CSV
df.to_csv('dataset_Final_FE.csv', index=False)

