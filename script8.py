#SCRIPT7 K-FOLD CROSS VALIDATION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifie
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pickle

url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final_Final.csv'
#url = 'dataset_Final_Final.csv'
df = pd.read_csv(url)

Y = df.CATEGORY
del df['CATEGORY']
X = df

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

cv = KFold(n_splits=10, shuffle=True, random_state=1)
results = list()

best_recall = 0
best_params = dict()
for train_ix, test_ix in cv.split(X_train):
 X_train_cv, X_test_cv = X_train.iloc[train_ix], X_train.iloc[test_ix]
 y_train_cv, y_test_cv = Y_train.iloc[train_ix], Y_train.iloc[test_ix]
 
 clf_dt = DecisionTreeClassifier()

 space = dict()
 space['max_depth'] = [16,17,18,19,20,21,22,23,24,25]
 space['criterion'] = ['gini','entropy']
 space['splitter'] = ['best','random']
 space['min_samples_split'] = [2,3,4,5]

 search = GridSearchCV(clf_dt, space, refit=True)
 result = search.fit(X_train_cv, y_train_cv)
 best_model = result.best_estimator_
 yhat = best_model.predict(X_test_cv)
 recall = metrics.recall_score(y_test_cv, yhat,average=None)
 print(recall)
 avg_recall = sum(recall)/5
 if avg_recall > best_recall:
   best_params = best_model.get_params()
   model = best_model

 print(avg_recall)
print(best_params)
print(type(model))
y_pred = model.predict(X_test)

print("Accuracy Final model:",metrics.accuracy_score(Y_test, y_pred))
print("Confusion Matrix final model : \n", metrics.confusion_matrix(Y_test, y_pred),"\n")
print("Precision final model:",metrics.precision_score(Y_test, y_pred,average=None))
print("Recall final model:",metrics.recall_score(Y_test, y_pred,average=None))

filename = 'final_model.sav'

pickle.dump(model, open( filename, "wb" ) )
