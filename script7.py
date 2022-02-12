#SCRIPT7 FEATURE SELECTION EVALUATION CORRELATION / DT IMPORTANCE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn import svm

#url = 'dataset_Final_FS_DT_1.csv'
url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final_FS_DT_1.csv'
#url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final_FS_COR_2.csv'

df = pd.read_csv(url)

Y = df.CATEGORY
del df['CATEGORY']
X = df

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
clf_svm = svm.SVC(kernel='linear') # Linear Kernel
clf_dt = DecisionTreeClassifier()

clf_svm.fit(X_train,y_train)
clf_dt.fit(X_train,y_train)

labels = clf_dt.classes_
labels2 = clf_svm.classes_
print(labels)
print(labels2)
print('======')

y_pred_dt=clf_dt.predict(X_test)
y_pred_svm=clf_svm.predict(X_test)

print("Accuracy for DT:",metrics.accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix Tree for DT : \n", metrics.confusion_matrix(y_test, y_pred_dt),"\n")
print("Precision for DT:",metrics.precision_score(y_test, y_pred_dt,average=None))
print("Recall for DT:",metrics.recall_score(y_test, y_pred_dt,average=None))

print("Accuracy for SVC:",metrics.accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix Tree for SVC : \n", metrics.confusion_matrix(y_test, y_pred_svm),"\n")
print("Precision for SVC:",metrics.precision_score(y_test, y_pred_svm,average=None))
print("Recall for SVC:",metrics.recall_score(y_test, y_pred_svm,average=None))