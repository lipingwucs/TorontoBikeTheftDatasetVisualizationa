# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:19:50 2020
@author: Group09
https://docs.google.com/presentation/d/e/2PACX-1vQA18l8l1xjQdvxVzzFHxfnMdgYl_mLlue-V2UlPsDO4MUPDqINxTYk71HFHteBwym5qOo9b-UB6fmL/pub?start=true&loop=false&delayms=3000
"""

'''
@author: Liping
'''
'''
2. Data modelling: 
2.1 Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.
2.2 Feature selection – use pandas and sci-kit learn.
2.3 Train, Test data splitting – use NumPy, sci-kit learn.
2.4 Managing imbalanced classes if needed.  Check here for info: https://elitedatascience.com/imbalanced-classes
'''


import pandas as pd
import os
import numpy as np
#change to your local 
path = "D:/CentennialWu/2020Fall/COMP309Data/GroupProject2/"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
print(fullpath)
df = pd.read_csv(fullpath)
pd.set_option('display.max_columns',20)
df.columns.values
df.shape   # 21584*30 columns
df.describe() 
df.describe
df.dtypes
df.head(5)

#

#check missing values of each column
df.isna().sum()

#For Bike model check how many null before fill the missing 
print(df['Bike_Model'].isnull().sum().sum()) #8140
df['Bike_Model'].fillna('UNKNOWN', inplace= True)
# check how many null after fill the missing 
print(df['Bike_Model'].isnull().sum())  #0

# For Bike_Colour, fill missing with "UNKNOWN"
# check how many null before fill the missing 
print(df['Bike_Colour'].isnull().sum()) #1729
df['Bike_Colour'].fillna('UNKNOWN', inplace= True)
# check how many null after fill the missing 
print(df['Bike_Colour'].isnull().sum())  #0

# fill missing Cost_of_Bike with median
median = df['Cost_of_Bike'].median()
# fill missing value or 0 with median 
df['Cost_of_Bike'].fillna(median, inplace= True)  
df['Cost_of_Bike'].replace(0, median, inplace= True)



'''
#Fill null for Column Status
print(df['Status'].isnull().sum().sum()) #
df['Status'].fillna( 1, inplace= True)
# check how many null after fill the missing 
print(df['Status'].isnull().sum()) 
df["Status"].value_counts()  # ['   ', 'RECOVERED', 'STOLEN', 'UNKNOWN']
# df_g9['Status'].replace(r'^\s*$', "UNKNOWN", regex=True)  # not working
df =df[~df["Status"].str.contains('   ')] 
df["Status"].value_counts()  # Working ['RECOVERED', 'STOLEN', 'UNKNOWN']
'''

# change 'Status' Column from Object to int
df['Status'] = df['Status'].map({'UNKNOWN':1,'STOLEN':1, '   ':1,'RECOVERED':0})
df.isna().sum()

df.columns
#from Occurrence_Date to get dayofweek
df["datetime"] = pd.to_datetime(df["Occurrence_Date"])
df['dayofweek'] =  df["datetime"].dt.dayofweek

# from Occurrence_Time to get hour of day
df["datehour"] = pd.to_datetime(df["Occurrence_Time"])
df['dayofhour'] =  df["datehour"].dt.hour
df['dayofhour'].dtype


# Drop some columns unwanted
df = df.drop(columns = ['X', 'Y', 'FID','Index_', 'event_unique_id','Occurrence_Date',"Occurrence_Time",'Neighbourhood','City','Location_Type',  'datehour', 'datetime'])
df = df.drop(columns = ['Primary_Offence'])
df = df.drop(columns = ['Occurrence_Year', 'Occurrence_Day'])
df = df.drop(columns = ['Division'])
df = df.drop(columns = ['dayofweek'])

df.columns.values
df.shape   #   (21584, 13)
df.describe() 
df.describe
df.dtypes
df.head(5)

'''
array(['Occurrence_Month', 'Premise_Type', 'Bike_Make', 'Bike_Model',
       'Bike_Type', 'Bike_Speed', 'Bike_Colour', 'Cost_of_Bike', 'Status',
       'Hood_ID', 'Lat', 'Long', 'dayofhour'], dtype=object)
'''

categories = []
for col, col_type in df.dtypes.iteritems():
     if col_type == 'O':
          categories.append(col)
     else:
          df[col].fillna(0, inplace=True)
print(categories)  # ['Premise_Type', 'Bike_Make', 'Bike_Model', 'Bike_Type', 'Bike_Colour']
print(df.columns.values)
print(df.head())
df.describe()
df.info()


# Convert hour of day to peaktime and none peaktime
#if hour is in [9, 12, 17, 18, 19] - Peaktime
# else non-peaktime
def IsPeakTime (hour):
    if hour in set([8,9, 12, 17, 18, 19,0]):
        return 1
    else:
        return 0

#print(df['dayofhour'][0])  
df['Peaktime'] =df['dayofhour']
for i in range(len(df['dayofhour'])):
    df['Peaktime'][i] = IsPeakTime( df['dayofhour'][i])
#print(df['Peaktime'][0])
for i in range(10):
    print(df['Peaktime'][i])
 #=====End of Peaktime Group
 
# Convert BikeColor into Black and NonBlack
def IsDark (color):
    if str(color).strip() in set(['BLK', 'BLU']):
        return 'Dark'
    else:
        return 'Light'

#print(df['ColorBlack'][0])  
df['ColorBlack'] =df['Bike_Colour']
for i in range(len(df['Bike_Colour'])):
    #print(df['Bike_Colour'][i])
    df['ColorBlack'][i] = IsDark(df['Bike_Colour'][i])

for i in range(len(df['ColorBlack'])):
    print(df['ColorBlack'][i])
#=====End of ColorBlack Group

# Occurrence_Month to Season
def ConvertToSeason (month):
    if month in set([6,7,8,9]):
        return 'Summer'
    elif  month in set([4,5,10,11]):
        return 'SpringOrFall'
    else:
        return 'Winter'
    
df['Season'] = df['Occurrence_Month']
for i in range(len(df['Occurrence_Month']) ):   
    df['Season'][i] = ConvertToSeason ( int(df['Occurrence_Month'][i]))
    
#===================== End of Season Convert
df.columns
'''
['Occurrence_Month', 'Premise_Type', 'Bike_Make', 'Bike_Model',
       'Bike_Type', 'Bike_Speed', 'Bike_Colour', 'Cost_of_Bike', 'Status',
       'Hood_ID', 'Lat', 'Long', 'dayofhour', 'Peaktime', 'ColorBlack',
       'Season'],
'''
#CostOfBike to CostRange
def GroupIntoRange (cost):
    if cost >1000:
        return "High"
    elif cost < 400:
        return "Low"
    else:
        return "Median"
    
df['CostRange'] = df['Cost_of_Bike']
type(df['Cost_of_Bike'][1])  #numpy.float64
for i in range(len(df['Cost_of_Bike'])  ):
    df['CostRange'][i] = GroupIntoRange (float(df['Cost_of_Bike'][i]))
print(df['CostRange'] )
#===================== End of Cost Convert to Price Range

#Graph and Visualization

#######PeakTime
 # Plot a histogram
import matplotlib.pyplot as plt
hist_DayofHour= plt.hist(df['Peaktime'],bins=2, width= 0.5)
plt.xlabel('PeaktimePeaktime')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Peaktime and Stolen')

import matplotlib.pyplot as plt
df['Peaktime'].value_counts()
df_peaktime = df['Peaktime'].value_counts()
#x= df_peaktime.index
x= ['PeakTime','NonePeakTime']
y = df_peaktime.values 
bar_peaktime= plt.bar(x, y)
#plt.xticks(rotation=90)
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- PeakTime and Stolen')
plt.show()

#############Seaspn
 # Plot a histogram
import matplotlib.pyplot as plt
hist_Season= plt.hist(df['Season'],bins=3, width= 0.5)
plt.xlabel('Season')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Season and Stolen')

import matplotlib.pyplot as plt
df['Season'].value_counts()
df_Season = df['Season'].value_counts()
x= df_Season.index
y = df_Season.values 
bar_peaktime= plt.bar(x, y)
plt.xticks(rotation=90)
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Season and Stolen')
plt.show()

###############BikeCost

 # Plot a histogram
import matplotlib.pyplot as plt
hist_Cost= plt.hist(df['CostRange'],bins=3, width= 0.5)
plt.xlabel('CostRange')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- CostRange and Stolen')

import matplotlib.pyplot as plt
df['CostRange'].value_counts()
df_CostRange = df['CostRange'].value_counts()
x= df_CostRange.index
y = df_CostRange.values 
bar_peaktime= plt.bar(x, y)
#plt.xticks(rotation=90)
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- CostRange and Stolen')
plt.show()

###############BikeColor ——》 ColorBlack
 # Plot a histogram
import matplotlib.pyplot as plt
hist_Color= plt.hist(df['ColorBlack'],bins=3, width= 0.5)
plt.xlabel('ColorBlack')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- ColorBlack and Stolen')

import matplotlib.pyplot as plt
df['ColorBlack'].value_counts()
df_ColorBlack = df['ColorBlack'].value_counts()
x= df_ColorBlack.index
y = df_ColorBlack.values 
bar_ColorBlack= plt.bar(x, y)
#plt.xticks(rotation=90)
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- ColorBlack and Stolen')
plt.show()

