# -*- coding: utf-8 -*-
"""
@author: Group 9 
2. Data modelling: 
2.1 Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.
2.2 Feature selection – use pandas and sci-kit learn.
2.3 Train, Test data splitting – use NumPy, sci-kit learn.

"""

# Import data
import pandas as pd
import os
import numpy as np
path = "C:/Users/User/Desktop/COMP309/group" # change to local path
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
bicycle_data = pd.read_csv(fullpath,sep=',')
print(bicycle_data.columns.values)

#Keep useful column
to_keep_list=['Status','Premise_Type','Hood_ID','Occurrence_Time','Occurrence_Month','Bike_Make','Bike_Colour','Cost_of_Bike']
bicycle_data_vars=bicycle_data.columns.values.tolist()
to_keep=[i for i in bicycle_data_vars if i in to_keep_list]
bicycle_data_final=bicycle_data[to_keep]
bicycle_data_final.columns.values

# Fill missing value 
bicycle_data_final.isna().sum()
print(bicycle_data['Cost_of_Bike'].mean()) # mean of cost of bike column
bicycle_data_final['Cost_of_Bike'].fillna(value=bicycle_data['Cost_of_Bike'].mean(), inplace=True)
bicycle_data_final['Bike_Colour'].fillna("Unknown", inplace=True)
bicycle_data_final.isna().sum()

# From Occurrence_Time to get hour of day
bicycle_data_final['Occurrence_Time'] = pd.to_datetime(bicycle_data_final['Occurrence_Time'])
bicycle_data_final['Occurrence_Time']=  bicycle_data_final['Occurrence_Time'].dt.hour

# Category Occurrence_Time to peak time 1 and unpeak time 0
def IsPeakTime (hour):
    if hour in set([9, 12, 17, 18, 19]):
        return 1
    else:
        return 0
for i in range(len(bicycle_data_final['Occurrence_Time'])):
    bicycle_data_final['Occurrence_Time'][i] = IsPeakTime(bicycle_data_final['Occurrence_Time'][i])


# Category make
def Make (make):
    if str(make).strip() == 'OT':
        return 'OT'
    if str(make).strip() == 'UK':
        return 'UK'
    if str(make).strip() == 'GI':
        return 'GI'
    if str(make).strip() == 'TR':
        return 'TR'
    if str(make).strip() == 'NO':
        return 'NO'
    if str(make).strip() == 'GIANT':
        return 'GIANT'
    if str(make).strip() == 'CC':
        return 'CC'
    else:
        return 'Other'

bicycle_data_final['Bike_Make'] =bicycle_data_final['Bike_Make']
for i in range(len(bicycle_data_final['Bike_Make'])):
   bicycle_data_final['Bike_Make'][i] = Make(bicycle_data_final['Bike_Make'][i])


# Convert BikeColor into Black and NonBlack
def IsDark (color):
    if str(color).strip() in set(['BLK', 'BLU']):
        return 'Dark'
    else:
        return 'Light'
 
bicycle_data_final['Bike_Colour'] =bicycle_data_final['Bike_Colour']
for i in range(len(bicycle_data_final['Bike_Colour'])):
   bicycle_data_final['Bike_Colour'][i] = IsDark(bicycle_data_final['Bike_Colour'][i])
bicycle_data_final['Bike_Colour']=( bicycle_data_final['Bike_Colour']=='Dark').astype(int)

# change 'Status' Column from Object to int
bicycle_data_final['Status'] = bicycle_data_final['Status'].map({'UNKNOWN':1,'STOLEN':1,'RECOVERED':0})
bicycle_data_final.isna().sum()
bicycle_data_final=bicycle_data_final.drop(index=0)

# Create dummy column for categorical column
cat_vars=['Premise_Type','Bike_Make']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(bicycle_data_final[var], prefix=var)
    bicycle_data_final_b1=bicycle_data_final.join(cat_list)
    bicycle_data_final=bicycle_data_final_b1

# Drop original categorical colum   
bicycle_data_final_vars=bicycle_data_final.columns.values.tolist()
to_keep=[i for i in bicycle_data_final_vars if i not in cat_vars]
bicycle_data_final=bicycle_data_final[to_keep]

# Prepare the data for the model build as X (inputs, predictor) and Y(output, predicted)
bicycle_data_final_vars=bicycle_data_final.columns.values.tolist()
Y=['Status']
X=[i for i in bicycle_data_final_vars if i not in Y ]
type(Y)
type(X)

X=bicycle_data_final[X]
Y=bicycle_data_final[Y]
type(Y)
type(X)

#Logistic regression split the data into 70% training and 30% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Decision tree split the data into 70% training and 30% for testing
bicycle_data_final_vars=bicycle_data_final.columns.values.tolist()
target=['Status']
predictors=[i for i in bicycle_data_final_vars if i not in target ]
DecisionX=bicycle_data_final[predictors]
DecisionY=bicycle_data_final[target]
#split the data sklearn module
from sklearn.model_selection import train_test_split
decisionTrainX,decisionTestX,decisionTrainY,decisionTestY = train_test_split(DecisionX,DecisionY, test_size = 0.3)


# Build the logistic regression model and validate the parameters
from sklearn import linear_model
from sklearn import metrics
logisticModel = linear_model.LogisticRegression(solver='lbfgs')
logisticModel.fit(X_train, Y_train)
#Run the test data against the new model
probs = logisticModel.predict_proba(X_test)
print(probs)
predicted = logisticModel.predict(X_test)
print (predicted)
print("Accuracy:",metrics.accuracy_score(Y_test, predicted))

# 10 fold cross validation using sklearn and all the data i.e validate the data	
from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), X, Y, scoring='accuracy', cv=10)
print (scores)
print ('Crossvalidation',scores.mean())


import numpy as np
prob=probs[:,1]
prob_df=pd.DataFrame(prob)
prob_df['predict']=np.where(prob_df[0]>=0.05,1,0)
Y_true =Y_test.values
Y_prob = np.array(prob_df['predict'])
preds=logisticModel.predict(X_test)
#Print the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_true, Y_prob)
print ("confusion_matrix\n",confusion_matrix)
#Import scikit-learn metrics module for accuracy, recall, f1 and precision score calculation
from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(Y_true, preds))
print("Recall:",metrics.recall_score(Y_true, preds))
print("F1 score:",metrics.f1_score(Y_true, preds))
print("Precision score:",metrics.precision_score(Y_true, preds))
#Calculate ROC
from sklearn.metrics import roc_auc_score
print("ROC-AUC:",roc_auc_score(Y_true,prob))



import joblib 
joblib.dump(logisticModel, 'C:/Users/User/Desktop/logisticRegressionModel.pkl')
print("Model dumped!")

model_columns = list(X.columns)
print(model_columns)
joblib.dump(model_columns, 'C:/Users/User/Desktop/logisticRegressionModel_columns.pkl')
print("Models columns dumped!")

# Build the decision tree model and validate the parameters
from sklearn.tree import DecisionTreeClassifier
decisionTreeModel = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_split=20, random_state=99)
decisionTreeModel.fit(decisionTrainX,decisionTrainY)
decisionPreds=decisionTreeModel.predict(decisionTestX)
decisionProbs = decisionTreeModel.predict_proba(decisionTestX)
decisionProb=decisionProbs[:,1]
print(decisionPreds)
print("Accuracy:",metrics.accuracy_score(decisionTestY, decisionPreds))

from sklearn.tree import export_graphviz
with open('C:/Users/User/Desktop/bicycle_dtree.dot', 'w') as dotfile:
    export_graphviz(decisionTreeModel, out_file = dotfile)
dotfile.close()

# 10 fold cross validation using sklearn and all the data i.e validate the data 
from sklearn.model_selection import KFold
#help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(decisionTreeModel,decisionTrainX, decisionTrainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print(score)

#Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(decisionTestY, decisionPreds))

from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(decisionTestY, decisionPreds))
print("Recall:",metrics.recall_score(decisionTestY, decisionPreds))
print("F1 score:",metrics.f1_score(decisionTestY,decisionPreds))
print("Precision score:",metrics.precision_score(decisionTestY, decisionPreds))
#Calculate ROC
from sklearn.metrics import roc_auc_score
print("AUC:",roc_auc_score(decisionTestY,decisionProb))

import joblib 
joblib.dump(decisionTreeModel, 'C:/Users/Users/Desktop/COMP309/group/decisionTreeModel.pkl')
print("Model dumped!")

decisionTreemodel_columns = list(DecisionX)
print(decisionTreemodel_columns)
joblib.dump(decisionTreemodel_columns, 'C:/Users/Users/Desktop/COMP309/group/decisionTreeModel_columns.pkl')
print("Models columns dumped!")


