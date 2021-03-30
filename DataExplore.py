# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:19:50 2020
@author: Group09
https://docs.google.com/presentation/d/e/2PACX-1vQA18l8l1xjQdvxVzzFHxfnMdgYl_mLlue-V2UlPsDO4MUPDqINxTYk71HFHteBwym5qOo9b-UB6fmL/pub?start=true&loop=false&delayms=3000
"""

'''
@author: Liping Wu
1. Data exploration: a complete review and analysis of the dataset 
1.1 Load the 'Bicycle_Thefts.csv' file into a dataframe and descibe data elements(columns),
 provide descriptions & types, ranges and values of elements as appropriate.
1.2 Statistical assessments including means, averages, correlations
1.3 Missing data evaluations – use pandas, NumPy and any other python packages
1.4 Graphs and visualizations – use pandas, matplotlib, seaborn, NumPy and any other python packages, you also can use power BI desktop.
'''

import pandas as pd
import os
import numpy as np
#change to your local 
path = "D:/CentennialWu/2020Fall/COMP309Data/GroupProject2/"
filename = 'Bicycle_Thefts.csv'
fullpath = os.path.join(path,filename)
print(fullpath)
data_bicycle = pd.read_csv(fullpath)
data_bicycle.columns.values
data_bicycle.shape   # 21584*30 columns
data_bicycle.describe() 
data_bicycle.describe
data_bicycle.dtypes
data_bicycle.head(5)
data_bicycle.tail(5)

#from Occurrence_Date to get dayofweek
data_bicycle["datetime"] = pd.to_datetime(data_bicycle["Occurrence_Date"])
data_bicycle['dayofweek'] =  data_bicycle["datetime"].dt.dayofweek
data_bicycle['dayofweek'].head()
type(data_bicycle['dayofweek'])

# from Occurrence_Time to get hour of day
data_bicycle["datehour"] = pd.to_datetime(data_bicycle["Occurrence_Time"])
data_bicycle['dayofhour'] =  data_bicycle["datehour"].dt.hour
data_bicycle['dayofhour'].head()
data_bicycle['dayofhour'].dtype


#drop unnecessary columns
df_g9 = data_bicycle.drop(columns = ['X', 'Y', 'FID','Index_', 'event_unique_id','Occurrence_Date',"Occurrence_Time",'Hood_ID','City', 'datehour', 'datetime'])
pd.set_option('display.max_columns',20)
print(df_g9.columns.values)
print(df_g9.shape)  # 19 columns
print(df_g9.describe())
print(df_g9.describe)
print(df_g9.dtypes) 
print(df_g9.head(5))   
print(df_g9.tail(5))   

# check column unique values
'''
print(df_g9.columns.values)

 'Division' 'Location_Type' 'Premise_Type' 'Bike_Make' 'Bike_Model'
 'Bike_Type' 'Bike_Speed' 'Bike_Colour' 'Cost_of_Bike' 'Status'
 'Neighbourhood' ]
'''

df_g9["Bike_Make"].nunique()
df_g9["Location_Type"].unique()

df_g9["Occurrence_Year"].value_counts()
df_g9["Occurrence_Year"].unique()

df_g9["Occurrence_Month"].value_counts()
df_g9["Occurrence_Month"].unique()

df_g9["Occurrence_Day"].value_counts()
df_g9["Occurrence_Day"].unique()

df_g9["dayofweek"].value_counts()
df_g9["dayofweek"].unique()

df_g9["dayofhour"].value_counts()
df_g9["dayofhour"].unique()

df_g9["Long"].value_counts()
df_g9["Lat"].unique()

df_g9["Neighbourhood"].value_counts()
df_g9["Neighbourhood"].unique()


#check null values of each column
df_g9.isna().sum()
# Or: 
print(len(df_g9)-df_g9.count())  #Only bike model and bike color and bike cost has some  null values

df_g9['Occurrence_Year'].unique()

#For Bike model check how many null before fill the missing 
print(df_g9['Bike_Model'].isnull().sum().sum()) #8140
df_g9['Bike_Model'].fillna('UNKNOWN', inplace= True)
# check how many null after fill the missing 
print(df_g9['Bike_Model'].isnull().sum())  #0

# For Bike_Colour, fill missing with "UNKNOWN"
# check how many null before fill the missing 
print(df_g9['Bike_Colour'].isnull().sum()) #1729
df_g9['Bike_Colour'].fillna('UNKNOWN', inplace= True)
# check how many null after fill the missing 
print(df_g9['Bike_Colour'].isnull().sum())  #0

# fill missing Cost_of_Bike with median
median = df_g9['Cost_of_Bike'].median()
print(median)
# fill missing value or 0 with median 
df_g9['Cost_of_Bike'].fillna(median, inplace= True)  
df_g9['Cost_of_Bike'].replace(0, median, inplace= True)
df_g9['Cost_of_Bike'].dtype #float
df_g9['Cost_of_Bike'].value_counts()

#check after (2 method)
print(df_g9['Cost_of_Bike'].isnull().sum())  #0 check after fill na will median
print(len(df_g9)-df_g9.count())  # check after replaced,0 all filled

##. group the data with bike feature
df_g9_bike = df_g9[['Bike_Make','Bike_Model', 'Bike_Type', 'Bike_Speed','Bike_Colour','Cost_of_Bike', 'Status']]
df_g9_bike.describe
# group the data with time 
# [(0-5:59), (6:00- 11:59), (12:00-5:59), (6-11:59) ]
df_g9_time = df_g9[['Occurrence_Year','Occurrence_Month', 'Occurrence_Day', 'dayofhour','dayofweek','Status']]
df_g9_time.describe
# One way to analysis stolen amount base hour of a day
Occurrence_Time_list = df_g9_time['Occurrence_Time']
def convertToTimeFrame(Occurrence_Time):
    hour = int(str(Occurrence_Time).split(':')[0])
    return int(hour/2) 
timeFrame =[0] *12
for x in Occurrence_Time_list:
    timeFrame[convertToTimeFrame(x)] +=1
print(timeFrame)  #[1727, 673, 447, 927, 2379, 1721, 2271, 2067, 2497, 2717, 2180, 1978]

# group the data with location information 
df_g9_location = df_g9[['Division','Neighbourhood', 'Premise_Type', 'Location_Type','Status']]
df_g9_location.columns.values
df_g9_location.describe
# group the data with geo information 
df_g9_geo = df_g9[['Lat', 'Long','Status']]
df_g9_geo.describe
print (df_g9_geo)   

# group the data with Primary_Offence
df_g9_offence = df_g9[['Primary_Offence', 'Status']]
df_g9_offence.describe
###########################################Geo Information
df_g9_geo = df_g9[['Lat', 'Long','Status']]

# Values and Labels:
df_g9["Status"].value_counts()  # ['   ', 'RECOVERED', 'STOLEN', 'UNKNOWN']
# df_g9['Status'].replace(r'^\s*$', "UNKNOWN", regex=True)  # not working
df_g9 =df_g9[~df_g9["Status"].str.contains('   ')] 
df_g9["Status"].value_counts()  # Working ['RECOVERED', 'STOLEN', 'UNKNOWN']
# group the data with Primary_Offence
df_g9_Status = df_g9['Status'].value_counts().head() # 21583 rows x 1 columns]

df_g9_Stolen = df_g9[df_g9["Status"].str.match('STOLEN')]  # 20928
df_g9_Recovered =df_g9[df_g9["Status"].str.contains('RECOVERED')] #252
df_g9_Unknown =df_g9[df_g9["Status"].str.match('UNKNOWN')]   #403

import pandas as pd
import matplotlib.pyplot as plt
labels = 'STOLEN', 'RECOVERED', 'UNKNOWN'
sizes = [20928, 252, 403]
explode = ( 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
colors = ['red', 'green', 'lightcoral']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors= colors,autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Toronto Bike Theft Status (2014-2019)")
plt.show()


### Draw a bike theft map 
#run in python cmd:    pip install folium
import matplotlib.pyplot as plt
import seaborn as snsm
import folium
import os
toronto_map = folium.Map(location=[43.6532,-79.3832],
                        zoom_start=11,
                        tiles="OpenStreetMap")
title_html = '''
             <h3 align="center" style="font-size:20px"><b>Toronto Bike Theft Map (2014-2019)</b></h3>
             '''
toronto_map.get_root().html.add_child(folium.Element(title_html))

#add Bike Status Stolen data to map
for i in range(len(df_g9_Stolen)):
    lat = (df_g9_Stolen['Lat'].array)[i]
    print(lat)
    long = (df_g9_Stolen['Long'].array)[i]
    print(long)
    folium.CircleMarker(location = [lat, long],ill = True, radius=1,color ='red').add_to(toronto_map)

#add Bike Status Recovered data to map
for i in range(len(df_g9_Recovered)):
    lat = (df_g9_Recovered['Lat'].array)[i]
    print(lat)
    long = (df_g9_Recovered['Long'].array)[i]
    print(long)
    folium.CircleMarker(location = [lat, long],ill = True, radius=1,color ='green').add_to(toronto_map)
    
#add Bike Status Unknown data to map
for i in range(len(df_g9_Unknown)):
    lat = (df_g9_Unknown['Lat'].array)[i]
    print(lat)
    long = (df_g9_Unknown['Long'].array)[i]
    print(long)
    folium.CircleMarker(location = [lat, long],ill = True, radius=1,color ='blue').add_to(toronto_map)
    
    
path ="D:/CentennialWu/2020Fall/COMP309Data/GroupProject2/" # change to your local path
mapfilename = "map_toronto_bike_status(2014-2019).html"
mapfullpath = os.path.join(path,mapfilename)
toronto_map.save(mapfullpath)
## End of drawing map

import matplotlib.pyplot as plt
df_g9.columns
# plot a scatter 
plt.scatter(df_g9['dayofweek'],df_g9['Cost_of_Bike'])
plt.show()
# plot a scatter 
plt.scatter(df_g9['Primary_Offence'],df_g9['Cost_of_Bike'])
plt.show()

######################  location factors   

#Note from Liping(for better informatipn)
#*************************HoodID

data_bicycle['Hood_ID'].value_counts()
#plot-hist
import matplotlib.pyplot as plt
hist_HoodId= plt.hist(data_bicycle['Hood_ID'],bins=18)
plt.xlabel('HoodId')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -HoodID vs Stolen')

#*************************HoodID

data_bicycle['Lat'].value_counts()
#plot-hist
import matplotlib.pyplot as plt
hist_HoodId= plt.hist(data_bicycle['Lat'],bins=18)
plt.xlabel('Lat')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -Lat vs Stolen')


data_bicycle['Long'].value_counts()
#plot-hist
import matplotlib.pyplot as plt
hist_HoodId= plt.hist(data_bicycle['Long'],bins=18)
plt.xlabel('Long')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -Long vs Stolen')

data_bicycle['X'].value_counts()
#plot-hist
import matplotlib.pyplot as plt
hist_HoodId= plt.hist(data_bicycle['X'],bins=18)
plt.xlabel('X')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -x vs Stolen')

data_bicycle['Y'].value_counts()
#plot-hist
import matplotlib.pyplot as plt
hist_HoodId= plt.hist(data_bicycle['Y'],bins=18)
plt.xlabel('Y')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -Y vs Stolen')


#*************************Division

df_g9["Division"].value_counts()  #18
df_g9["Division"].value_counts().head(5) 
df_g9["Division"].value_counts().tail(5) 
df_g9["Division"].value_counts().keys() 
df_g9['Division'].describe()
df_g9['Division'].unique()
df_g9_division_top5 =df_g9["Division"].value_counts().head(5) 
df_g9_division_safest5 =df_g9["Division"].value_counts().tail(5) 

# plot a scatter 
plt.scatter(df_g9['Division'],df_g9['Cost_of_Bike'])
plt.show()

#plot-hist
import matplotlib.pyplot as plt
hist_Division= plt.hist(df_g9['Division'],bins=18)
plt.xlabel('Division')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -Division vs Stolen')

#plot-bar-Top 5 Division
import matplotlib.pyplot as plt
# x = df_g9_division_top5.index #[52, 14, 51, 53, 55]
x= ['52', '14', '51', '53', '55']#[3913, 3845, 3572, 1748, 1614]
y = df_g9_division_top5.values  
bar_Division = plt.bar(x,y, width=0.5, color ='red')
plt.xlabel('Division')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)   Top5 Division ')

#plot-bar-Safest 5 Division
import matplotlib.pyplot as plt
# x = df_g9_division_safest5.index #[33, 12, 23, 42, 58]
x= ['33', '12', '23', '42', '58']#[3913, 3845, 3572, 1748, 1614]
y = df_g9_division_safest5.values  
bar_Division = plt.bar(x,y, width=0.5, color ='green')
plt.xlabel('Division')
plt.ylabel('Stolen')
plt.title(' Toronto Bike Theft (2014-2019) Safest 5  Division')
#*************************Neighbourhood

df_g9["Neighbourhood"].value_counts()  #140
df_g9_NeighbourhoodTop10 = df_g9["Neighbourhood"].value_counts().head(10) 
df_g9_NeighbourhoodSafest10 = df_g9["Neighbourhood"].value_counts().tail(10) 
df_g9["Neighbourhood"].value_counts().keys()
df_g9['Neighbourhood'].describe()
df_g9['Neighbourhood'].unique()

#plot-hist
import matplotlib.pyplot as plt
hist_Neighbourhood= plt.hist(df_g9['Neighbourhood'],bins=140)
plt.xticks(rotation=90)
plt.xlabel('Neighbourhood')
plt.ylabel('Stolen')
plt.title(' Toronto Bike Theft (2014-2019) Neighbourhood and Stolen')

#plot-bar-Top 10 Neighbourhood
import matplotlib.pyplot as plt
x = df_g9_NeighbourhoodTop10.index  
y = df_g9_NeighbourhoodTop10.values  
bar_Division = plt.bar(x,y, width=0.5, color ='red')
plt.xticks(rotation=90)
plt.xlabel('Neighbourhood')
plt.ylabel('Stolen')
plt.title(' Toronto Bike Theft (2014-2019) -Top10 Neighbourhood  and Stolen')

#plot-bar-Safest 10 Neighbourhood
import matplotlib.pyplot as plt
x = df_g9_NeighbourhoodSafest10.index 
y = df_g9_NeighbourhoodSafest10.values  
bar_Division = plt.bar(x,y, width=0.5, color ='green')
plt.xticks(rotation=90)
plt.xlabel('Neighbourhood')
plt.ylabel('Stolen')
plt.title(' Toronto Bike Theft (2014-2019) -  Safest 10 Neighbourhood and Stolen')

#*************************Premise_Type

df_g9['Premise_Type'].value_counts()  # 5 type

 # Plot a histogram
import matplotlib.pyplot as plt
hist_Premise= plt.hist(df_g9['Premise_Type'],bins=12)
plt.xlabel('Premise_Type')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -  Premise_Type and Stolen')

#plot-bar- Premise_Type (from high to low)
import matplotlib.pyplot as plt
df_g9_Premise =df_g9['Premise_Type'].value_counts().head()
x = df_g9_Premise.index 
y = df_g9_Premise.values  
bar_Division = plt.bar(x,y, width=0.5, color ='red')
plt.xticks(rotation=45)
plt.xlabel('Premise Type')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019) -  Premise_Type  and Stolen')

#************************* Location_Type

df_g9['Location_Type'].value_counts()  # 44 ypes
df_g9_LocationTop10 = df_g9['Location_Type'].value_counts().head(10)
df_g9_LocationSafe10 = df_g9['Location_Type'].value_counts().tail(10)
df_g9['Location_Type'].describe()
df_g9['Location_Type'].unique()

 # Plot a histogram
import matplotlib.pyplot as plt
hist_Location_Type= plt.hist(df_g9['Location_Type'],bins=44)
plt.xticks(rotation=90)
plt.xlabel('Location_Type')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)-Location_Type and Stolen') 

#plot-bar-Top 10 Neighbourhood
import matplotlib.pyplot as plt
x = df_g9_LocationTop10.index  
y = df_g9_LocationTop10.values  
bar_Location_Type = plt.bar(x,y, width=0.5, color ='red')
plt.xticks(rotation=90)
plt.xlabel('Location_Type')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)-Location_Type Top10 and Stolen')

#plot-bar-Safest 10 Neighbourhood
import matplotlib.pyplot as plt
x = df_g9_LocationSafe10.index 
y = df_g9_LocationSafe10.values  
bar_Location_Type = plt.bar(x,y, width=0.5, color ='green')
plt.xticks(rotation=90)
plt.xlabel('Location_Type')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)-Location_Type Safest 10 and Stolen')


#########################bike feature


#************************* Bike_Make

df_g9['Bike_Make'].value_counts() # 725
df_g9_BikeMakeTop10 =df_g9['Bike_Make'].value_counts().head(10)
df_g9_BikeMakeBottom10 = df_g9['Bike_Make'].value_counts().tail(10)
df_g9['Bike_Make'].describe()
df_g9['Bike_Make'].unique()

# Plot a histogram
import matplotlib.pyplot as plt
hist_Bike_Make= plt.hist(df_g9['Bike_Make'],bins=725)
plt.xticks(rotation=90)
plt.xlabel('Bike_Make')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Bike_Make and Stolen') 

#plot-bar-Top 10 Bike_Make
import matplotlib.pyplot as plt
x = df_g9_BikeMakeTop10.index  
y = df_g9_BikeMakeTop10.values  
bar_Bike_Make = plt.bar(x,y, width=0.5, color ='red')
plt.xticks(rotation=90)
plt.xlabel('Bike_Make')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)-Bike_Make Top10 and Stolen')

#plot-bar-Safest 10 Bike_Make
import matplotlib.pyplot as plt
x = df_g9_BikeMakeBottom10.index 
y = df_g9_BikeMakeBottom10.values  
bar_Bike_Make = plt.bar(x,y, width=0.5, color ='green')
plt.xticks(rotation=90)
plt.xlabel('Bike_Make')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Bike_Make Safest 10 and Stolen')

#*************************Bike_Colour

df_g9['Bike_Colour'].value_counts() #233
df_g9_BikeColorTop10 = df_g9['Bike_Colour'].value_counts().head(10)
df_g9_BikeColorBottom10 = df_g9['Bike_Colour'].value_counts().tail(10)
df_g9['Bike_Colour'].describe()
df_g9['Bike_Colour'].unique()

#plot-bar-Top 10 Bike_Colour
import matplotlib.pyplot as plt
x = df_g9_BikeColorTop10.index  
y = df_g9_BikeColorTop10.values  
bar_Bike_Make = plt.bar(x,y, width=0.5, color ='red')
plt.xticks(rotation=90)
plt.xlabel('Bike_Colour')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Bike_Colour Top10 and Stolen')

#plot-bar-Safest 10 Bike_Colour
import matplotlib.pyplot as plt
x = df_g9_BikeColorBottom10.index 
y = df_g9_BikeColorBottom10.values  
bar_Bike_Make = plt.bar(x,y, width=0.5, color ='green')
plt.xticks(rotation=90)
plt.xlabel('Bike_Colour')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)-Bike_Colour Safest 10 and Stolen')

#************************* Cost_of_Bike
df_g9['Cost_of_Bike'].describe()
df_g9['Cost_of_Bike'].unique()

df_g9['Cost_of_Bike'].value_counts() #1458
df_g9_BikeCostTop5 = df_g9['Cost_of_Bike'].value_counts().head(5)
df_g9['Cost_of_Bike'].describe()
df_g9['Cost_of_Bike'].unique()

# Plot a boxplot
import matplotlib.pyplot as plt
boxplot_BikeCost= plt.boxplot(df_g9['Cost_of_Bike'])
plt.ylim(0,4000)
plt.ylabel('Cost of Bike')
plt.title('Toronto Bike Theft (2014-2019)- Bike_Cost Range and Stolen')
plt.show()

hist_BikeCost= plt.hist(df_g9['Cost_of_Bike'])
plt.xticks(np.arange(0, 2000, step=1000))
plt.xlabel('Cost of Bike')
plt.show()


#plot-bar-Top 5 BikeCost
import matplotlib.pyplot as plt
# x = df_g9_BikeCostTop5.index  # 600.0, 500.00, 1000.0, 800.0, 400.0
x =[ '600.0','500.0', '1000.0', '800.0', '400']
y = df_g9_BikeCostTop5.values   #[1682, 1200, 1153,  979,  877]
bar_Bike_Make = plt.bar(x,y, width=0.5, color ='red')
plt.xticks(rotation=90)
plt.xlabel(' BikeCost Top5')
plt.ylabel('Stolen')
plt.title(' Toronto Bike Theft (2014-2019)-Bike Cost Top5 and stolen')



#************************* Bike_Type

df_g9['Bike_Type'].value_counts() #13
df_g9_BikeTypeTop5 = df_g9['Bike_Type'].value_counts().head(5)
df_g9_BikeTypeBottom5 = df_g9['Bike_Type'].value_counts().tail(5)
df_g9['Bike_Type'].value_counts().tail()
df_g9['Bike_Type'].unique()
df_g9['Bike_Type'].describe()
df_g9['Bike_Type'].unique()

# Plot a histogram
import matplotlib.pyplot as plt
hist_Bike_Type= plt.hist(df_g9['Bike_Type'],width= 0.5, bins=13)
plt.xticks(rotation=90)
plt.xlabel('Bike_Type')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Bike_Type and Stolen') 

#plot-bar-Top 5 Bike_Type
import matplotlib.pyplot as plt
x = df_g9_BikeTypeTop5.index  
y = df_g9_BikeTypeTop5.values  
bar_Bike_Make = plt.bar(x,y, width=0.5, color ='red')
#plt.xticks(rotation=90)
plt.xlabel('Bike_Type')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Bike_Type Top5 and stolen')

#Bike Model
hist_Bike_Model = plt.hist(df_g9['Bike_Model'])
df_g9['Cost_of_Bike' ].value_counts()

#plot-bar-Bottom 5 Bike_Type
import matplotlib.pyplot as plt
x = df_g9_BikeTypeBottom5.index  
y = df_g9_BikeTypeBottom5.values  
bar_Bike_Make = plt.bar(x,y, width=0.5, color ='green')
#plt.xticks(rotation=90)
plt.xlabel('Bike_Type')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Bike_Type Bottom 5 and stolen')

#*************************

df_g9['Bike_Speed'].value_counts() #62
df_g9_BikeSpeedTop5  = df_g9['Bike_Speed'].value_counts().head(5)
df_g9['Bike_Speed'].value_counts().tail()
df_g9['Bike_Speed'].unique()
df_g9['Bike_Speed'].describe()

import matplotlib.pyplot as plt
#x= df_g9_BikeSpeedTop5.index
x= ['21', '1', '18', '24', '10']
y= df_g9_BikeSpeedTop5.values 
width =0.35
hist_Division= plt.bar(x, y,width,color='red') # rgb color=(0.9, 0.2, 0.2, 0.6)
plt.xlabel("Speed")
plt.ylabel("Stolen")
#plt.xticks(rotation=90)
plt.title('Toronto Bike Theft (2014-2019)-Top5 Bike_Speed Stolen')
plt.show()

###['Occurrence_Year','Occurrence_Month', 'Occurrence_Day', 'Dayofhour','dayofweek',
#*************************
df_g9['Occurrence_Year'].value_counts() #6
df_g9_year = df_g9['Occurrence_Year'].value_counts().head(6)
#Bar plot
import matplotlib.pyplot as plt
print(df_g9['Occurrence_Year'].unique())
x = df_g9_year.index #[2018, 2017, 2016, 2019, 2015, 2014]
y =df_g9_year.values   #[3949, 3863, 3800, 3673, 3285, 3014]
hist_year= plt.bar(x,y,color ='red', width =0.35)
plt.xlabel('Year')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Year and Stolen')
plt.show()

#*************************Month
df_g9['Occurrence_Month'].value_counts() #12
df_g9_month = df_g9['Occurrence_Month'].value_counts().head(12)
import matplotlib.pyplot as plt
#x= df_g9_month.index #[7, 8, 6, 9, 5, 10, 4, 11, 12, 3, 1, 2]
x =['7', '8', '6', '9', '5', '10', '4', '11', '12', '3',' 1', '2']
y = df_g9_month.values 
 #[3358, 3052, 3006, 2763, 2348, 2113, 1310, 1252,  726,  684,  525, 447]
hist_month= plt.bar(x, y,color ='red', width =0.35)
plt.xlabel('Month')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Month and Stolen (From high to low) ')
plt.show() 

# same trend
import matplotlib.pyplot as plt
hist_month= plt.hist(df_g9['Occurrence_Month'],bins=12,width =0.35)
plt.xlabel('Occurrence_Month')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Month (1-12) and Stolen')

#*************************Day
df_g9['Occurrence_Day'].value_counts() #30
df_g9_day =df_g9['Occurrence_Day'].value_counts()
import matplotlib.pyplot as plt
x =  df_g9_day.index
y = df_g9_day.values
hist_month= plt.bar(x, y,align='edge')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- DayofMonth and Stolen')
plt.show()


#*************************Day Of Week
import matplotlib.pyplot as plt
df_g9_dayofweek = df_g9['dayofweek'].value_counts() #7
df_g9['dayofweek'].value_counts() #24
df_g9['dayofweek'].value_counts().head(4)
df_g9['dayofweek'].value_counts().tail(4)


#x = df_g9_dayofweek.index # [4, 2, 3, 0, 1, 5, 6]
x =['Thu','Tue', 'Wed','Sun', 'Mon', 'Fri', 'Sat']
y = df_g9_dayofweek.values
hist_DayofWeek= plt.bar(x, y,color ='red' , width= 0.5)
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- DayOfWeek and Stolen (from high to low)')
plt.show()

#*************************Hour of Day
df_g9['dayofhour'].value_counts() #30
df_g9_hour=df_g9['dayofhour'].value_counts().head()
import matplotlib.pyplot as plt
# x =  df_g9_hour.index  #[18, 17, 12, 9, 19]
x= ['18', '17', '12', '9', '19']
y = df_g9_hour.values
hist_month= plt.bar(x, y,align='center', color ='red' , width= 0.5)
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Hour of a Day (Top 5) and Stolen')
plt.show()



 # Plot a histogram
import matplotlib.pyplot as plt
hist_DayofHour= plt.hist(df_g9['dayofhour'],bins=24, width= 0.5)
plt.xlabel('HourOfDay')
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- HourOfDay and Stolen')

#************************* Primary_Offence
### 'Primary_Offence'
import matplotlib.pyplot as plt

df_g9['Primary_Offence'].value_counts() #65
df_g9['Primary_Offence'].value_counts().tail(10)
df_g9_offence_top5 = df_g9['Primary_Offence'].value_counts().head(5)

x= df_g9_offence_top5.index
y = df_g9_offence_top5.values 
hist_DayofWeek= plt.bar(x, y)
plt.xticks(rotation=90)
plt.ylabel('Stolen')
plt.title('Toronto Bike Theft (2014-2019)- Offence Top 5 and Stolen')
plt.show()





