#SCRIPT3 TEST DT/SVC WITH SMOTE/CLASS WEIGHTS/NOTHING
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn import metrics
from sklearn import svm
from imblearn.over_sampling import SMOTE 



url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/dataset_Final_FE.csv'
#url = 'dataset_Final_FE.csv'

df = pd.read_csv(url)
#KEEP LABELS 
Y = df.CATEGORY
del df['CATEGORY']
del df['class']
del df['num_outbound_cmds']
X = df
print("Starting")
#TEST THREE SCENARIOS
for i in range(3):
  if i == 0:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    clf_svm = svm.SVC(kernel='linear') # Linear Kernel
    clf_dt = DecisionTreeClassifier()

    clf_svm.fit(X_train,y_train)
    clf_dt.fit(X_train,y_train)

    y_pred_dt=clf_dt.predict(X_test)
    y_pred_svm=clf_svm.predict(X_test)

    print("Accuracy for DT:",metrics.accuracy_score(y_test, y_pred_dt))
    print("Confusion Matrix Tree for DT : \n", metrics.confusion_matrix(y_test, y_pred_dt),"\n")
    print("Precision for DT:",metrics.precision_score(y_test, y_pred_dt,average=None))
    print("Recall for DT:",metrics.recall_score(y_test, y_pred_dt,average=None))

    print("Accuracy for SVM:",metrics.accuracy_score(y_test, y_pred_svm))
    print("Confusion Matrix Tree for SVM : \n", metrics.confusion_matrix(y_test, y_pred_svm),"\n")
    print("Precision for SVM:",metrics.precision_score(y_test, y_pred_svm,average=None))
    print("Recall for SVM:",metrics.recall_score(y_test, y_pred_svm,average=None))

  elif i == 1:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    clf_svm = svm.SVC(kernel='linear',class_weight='balanced') # Linear Kernel
    clf_dt = DecisionTreeClassifier(class_weight='balanced')

    clf_svm.fit(X_train,y_train)
    clf_dt.fit(X_train,y_train)

    y_pred_dt=clf_dt.predict(X_test)
    y_pred_svm=clf_svm.predict(X_test)

    print("Accuracy with class weights for DT:",metrics.accuracy_score(y_test, y_pred_dt))
    print("Confusion Matrix Tree with class weights for DT : \n", metrics.confusion_matrix(y_test, y_pred_dt),"\n")
    print("Precision with class weights for DT:",metrics.precision_score(y_test, y_pred_dt,average=None))
    print("Recall with class weights for DT:",metrics.recall_score(y_test, y_pred_dt,average=None))

    print("Accuracy with class weights for SVM:",metrics.accuracy_score(y_test, y_pred_svm))
    print("Confusion Matrix with class weights Tree for SVM : \n", metrics.confusion_matrix(y_test, y_pred_svm),"\n")
    print("Precision with class weights for SVM:",metrics.precision_score(y_test, y_pred_svm,average=None))
    print("Recall with class weights for SVM:",metrics.recall_score(y_test, y_pred_svm,average=None))

  else:

    smi = SMOTE(random_state=42)
    X_res, Y_res = smi.fit_resample(X, Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    clf_svm = svm.SVC(kernel='linear') # Linear Kernel
    clf_dt = DecisionTreeClassifier(class_weight='balanced')

    clf_svm.fit(X_train,y_train)
    clf_dt.fit(X_train,y_train)

    y_pred_dt=clf_dt.predict(X_test)
    y_pred_svm=clf_svm.predict(X_test)

    print("Accuracy with SMOTE for DT:",metrics.accuracy_score(y_test, y_pred_dt))
    print("Confusion Matrix with SMOTE Tree for DT : \n", metrics.confusion_matrix(y_test, y_pred_dt),"\n")
    print("Precision with SMOTE for DT:",metrics.precision_score(y_test, y_pred_dt,average=None))
    print("Recall with SMOTE for DT:",metrics.recall_score(y_test, y_pred_dt,average=None))

    print("Accuracy with SMOTE for SVM:",metrics.accuracy_score(y_test, y_pred_svm))
    print("Confusion with SMOTE Matrix Tree for SVM : \n", metrics.confusion_matrix(y_test, y_pred_svm),"\n")
    print("Precision with SMOTE for SVM:",metrics.precision_score(y_test, y_pred_svm,average=None))
    print("Recall with SMOTE for SVM:",metrics.recall_score(y_test, y_pred_svm,average=None))
  
