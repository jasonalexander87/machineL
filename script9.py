#SCRIPT9 SAVED MODEL ON ACTION
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
import pickle

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#url = 'samples_Final.csv'
url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/samples_Final.csv'
df = pd.read_csv(url)

Y = df.CATEGORY
del df['CATEGORY']
X = df

y_pred = loaded_model.predict(X)

print("Accuracy for DT:",metrics.accuracy_score(Y, y_pred))
print("Confusion Matrix Tree for DT : \n", metrics.confusion_matrix(Y, y_pred),"\n")
print("Precision for DT:",metrics.precision_score(Y, y_pred,average=None))
print("Recall for DT:",metrics.recall_score(Y, y_pred,average=None))


