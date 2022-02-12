#SCRIPT10 MODEL TESTING SPEED EVALUATION
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifie
import pickle
import time

url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final_Final.csv'
#url = 'dataset_Final_Final.csv'
df = pd.read_csv(url)

Y = df.CATEGORY
del df['CATEGORY']
X = df

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

start = time.time()

y_pred = loaded_model.predict(X)

end = time.time() - start
print(end,'seconds')
print(' for ',df.shape[0],'samples')