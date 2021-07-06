#!/usr/bin/env python
# coding: utf-8

# # BUILDING THE MODEL TO FORECAST THE UPCOMING TOTAL NO
# # OF CONFIRMED CASES IN NEXT 10 DAYS
# 

# In[1]:


import pandas as pd
import numpy as np
import random
import math
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
import operator
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


conf_case=pd.read_csv("D:\\RStudio\\mini COVID\\MODEL FOR COVID\\DATA SET UPDATED\\cleaned/time_series_covid_19_confirmed_US.csv")


# In[3]:


death_case=pd.read_csv("D:\\RStudio\\mini COVID\\MODEL FOR COVID\\DATA SET UPDATED\\cleaned/time_series_covid_19_deaths_US.csv")


# In[4]:


#first 10 value
conf_case.head(10)


# In[5]:


#first 10 value 
death_case.head(10)


# In[ ]:





# In[6]:


conf_case.columns


# In[7]:


death_case.columns


# In[ ]:





# In[8]:


# SVM AND LINEAR REGRESSION FOR PREDTION OF TOTAL NO OF  CONFIRMED CASE 
# MAY COME UP NEXT  DAYS(10)


# In[ ]:





# In[9]:


# EXTRACT all OF THE COLUMN 
cols=conf_case.keys()
cols
#now we have all the column name snd the index of each in the cols


# In[10]:


#extract only the date from the conf and death case
conf=conf_case.loc[:,cols[11]:cols[-1]]
death=death_case.loc[:,cols[11]:cols[-1]]


# In[11]:


print("---------------------------CONFIRMED-----------------------------------")
print("\n")
print(conf.columns)#we need only date date for futher use so extrated
print("\n------------------------------DEATH---------------------------------")
print(death.columns)#we need only date date for futher use so extrated


# In[12]:


#here we have the total date of 114 days
#checking the data of OUT BREAKCASE
conf.head()


# In[13]:


#finding the total conf case,and death cases and appending to the list

#now since we have the death and confriemed case data we can find the total 
#mortality rare which is the death_sum/conf_sum


# In[14]:


dates=conf.keys()
usa_cases=[]
tot_death=[]
mortality_rate=[]

for i in dates:
    conf_sum=conf[i].sum()
    death_sum=death[i].sum()
    tot_death.append(death_sum)
    usa_cases.append(conf_sum)
    mortality_rate.append(death_sum/conf_sum)


# In[15]:


#display the new variable
print("\n-----------------------------------------------------------------")
print("\n","THE TOTAL CONFIREMD CASE IS :",conf_sum)
print("\n","THE TOTAL DEATH CASES :",death_sum)
print("\n","THE TOTAL USA CASES GROWTH DATE WISE :",usa_cases)
print("\n-----------------------------------------------------------------")
#list(mortality_rate)


# In[16]:


#After 62 DAys there is growth on the confirmed case al most 10000
#IT IS THE array is listed with date wise ALL CONFRIMED CASE


# In[47]:


#CONVERTED THE DATE AND THE CASE IN FORM OF NUMPY ARRAY 
date_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
usa_cases=np.array(usa_cases).reshape(-1,1)
tot_death=np.array(tot_death).reshape(-1,1)


# In[54]:


#taking the dATA AFTER THE 101 conf case

u_case=usa_cases[usa_cases>100]
d_sin=date_since_1_22[date_since_1_22>=42]

u_case=np.array(u_case).reshape(-1,1)
d_sin=np.array(d_sin).reshape(-1,1)

print(" TOTAL USA CONF CASE AFTER 101 death: ",len(u_case))
print(len(d_sin))


# In[18]:


date_since_1_22#array of the total days 144 strating from the 1 jan to 14 may


# In[19]:


usa_cases#array of total no of case emerge in the usa from the strat date


# In[20]:


tot_death#array of total no of death_case emerge in the usa from the strat date


# In[21]:


#future foescasting for next days
#i have add the 10 days more to the total number of the days
dif=10
future_forecast=np.array([i for i in range(len(dates)+dif)]).reshape(-1,1)
adjusted_dates=future_forecast[:-10]


# In[22]:


future_forecast#now it has become 113 to (113+10 days further=123 for future forecasting)


# In[23]:


#converting the int value into date time value for better visualization

start='1/22/2020'
start_date=datetime.datetime.strptime(start,'%m/%d/%Y')
future_forecast_dates=[]
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    


# In[24]:


future_forecast_dates#printing in the date in the perfect format format 


# In[25]:


#visualization with latest date of 14may

latest_conf=conf_case[dates[-1]]
latest_death=death_case[dates[-1]]


# In[26]:


latest_conf #last column value for all dataset giving yoiu the conf case across various regiion


# In[27]:


latest_death #last column value for all dataset giving yoiu the DEATH case across various regiion


# In[28]:


#unique regiions or country list 
unique_prov=list(conf_case['Province_State'].unique())
unique_prov


# In[29]:


#calu total con case in us
prov_conf_cases=[]
no_cases=[]
for i in unique_prov:
    cases=latest_conf[conf_case['Province_State']==i].sum()
    if cases>0:
        prov_conf_cases.append(cases)
    else:
        no_cases.append(i)

for i in no_cases:
    unique_prov.remove(i)
    


# In[30]:


#now no of case according to the per provi/state/city

for i in range(len(unique_prov)):
    print(f'\t{unique_prov[i]}:\t{prov_conf_cases[i]}  cases')


# In[31]:


#handlind the na value if any
nan_ind=[]

for i in range(len(unique_prov)):
    if type(unique_prov[i])==float:
        nan_ind.append(i)

unique_prov=list(unique_prov)
prov_conf_cases=list(prov_conf_cases)

for i in nan_ind:
    unique_prov.pop(i)
    prov_conf_cases.pop(i)


# In[32]:


plt.figure(figsize=(32,32))
plt.bar(unique_prov,prov_conf_cases)
plt.title("NUMBER OF COVID-19 CONF CASES IN USA")
plt.xlabel("NUMBER OF COVID-19 CONF CASES IN USA")


# In[33]:


ny_conf=latest_conf[conf_case['Province_State']=="New York"].sum()
all_other=np.sum(prov_conf_cases)-ny_conf
plt.figure(figsize=(16,9))
plt.barh("NEW YORK",ny_conf)
plt.barh("ALL OTHER",all_other)
plt.title("NO OF CONFIRMED CASE")
plt.show()


# In[80]:


print("--------------------------------------------------------")
print("| NEW YORK CASES:\t|\t\t",format(ny_conf),"\t|")
print("| ALL OTHER STATE:\t|\t\t",format(all_other),"\t|")
print("| TOTAL: \t\t|\t\t",format(ny_conf+all_other),"\t|")
print("--------------------------------------------------------")


# In[35]:


# less no of conf in the other and the top ten state with higher no of conf
vi_unique_state=[]
vi_conf_cases=[]
other=np.sum(prov_conf_cases[10:])
for i in range(len(prov_conf_cases[:10])):
    vi_unique_state.append(unique_prov[i])
    vi_conf_cases.append(prov_conf_cases[i])
    
vi_unique_state.append("Others")
vi_conf_cases.append("others")
     


# In[38]:


print("\n\n",list(vi_unique_state))
print("\n\n",list(vi_conf_cases))


# In[39]:


#ploting


# #  BUILDING THE ""SVM"" MODEL TO PREDICT THE UPCOMING TOTAL NO 
# 
# # OF CONFIRMED CASES IN NEXT 10 DAYS

# In[40]:


from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[55]:


#X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(date_since_1_22, usa_cases, test_size=0.15, shuffle=False)

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(d_sin, u_case, test_size=0.15, shuffle=False)


# In[59]:


y_test_confirmed


# In[60]:


kernel=['poly','sigmoid','rbf']#by default rbf is used
c=[0.01,0.1,1,10]
gamma=[0.01,0.1,1]
epsilon=[0.01,0.1,1]
shrinking=[True, False]

svm_grid={'kernel':kernel,'C':c,'gamma':gamma,'epsilon':epsilon,'shrinking':shrinking}


svm=SVR()
svm_search=RandomizedSearchCV(svm,svm_grid,scoring='neg_mean_squared_error',cv=3,return_train_score=True,n_jobs=-1,n_iter=40,verbose=1)
svm_search.fit(X_train_confirmed,y_train_confirmed)


# In[61]:


svm_search.best_params_


# In[62]:


svm_conf=svm_search.best_estimator_
svm_pred=svm_conf.predict(future_forecast)


# In[63]:


svm_conf#to find the best estimator


# In[64]:


list(svm_pred)# pred value now we have to test again the testing data


# In[65]:


#check against testing data

svm_test_pred=svm_conf.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)

print("MAE:",mean_absolute_error(svm_test_pred,y_test_confirmed))
print("MSE:",mean_squared_error(svm_test_pred,y_test_confirmed))


# In[66]:


#total no of coro cases time

plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,usa_cases)
plt.title("NO OF CORONA CASE ",size=30)
plt.xlabel("DAYS SINCE 1/22/2020",size=30)
plt.ylabel("NO OF CASES",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
#plt.legend(['Confirmed cases','SVM_prediction'])


# # y axis total no of CASES
# 
# # x axis total no of days
# 
# # after the 62 days the rate of confirmed cases went HIGH
# 

# In[67]:


#total no of coFIRMED CASES VS THE PREDICTED cases time

plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,usa_cases)
plt.plot(future_forecast,svm_pred,linestyle="dashed",color='purple')
plt.title("NO OF CORONA CASE ",size=30)
plt.xlabel("DAYS SINCE 1/22/2020",size=30)
plt.ylabel("NO OF CASES",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(['Confirmed cases','SVM_prediction'])


# In[68]:


print("SVM FUTURE PREDTICTION:")
set(zip(future_forecast_dates[-10:],svm_pred[-10:]))


# #  BUILDING THE LINEAR MODEL TO PREDICT THE UPCOMING TOTAL NO 
# 
# # OF CONFIRMED CASES IN NEXT 10 DAYS

# In[69]:


from sklearn.linear_model import LinearRegression
linear_model=LinearRegression(normalize=True,fit_intercept=True)
linear_model.fit(X_train_confirmed,y_train_confirmed)
test_linear_pred=linear_model.predict(X_test_confirmed)
linear_pred=linear_model.predict(future_forecast)

print("MAE:",mean_absolute_error(svm_test_pred,y_test_confirmed))
print("MSE:",mean_squared_error(svm_test_pred,y_test_confirmed))


# In[70]:


linear_pred


# In[71]:


plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)


# In[72]:


plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,usa_cases)
plt.plot(future_forecast,linear_pred,linestyle="dashed",color='purple')
plt.title("NO OF CORONA CASE ",size=30)
plt.xlabel("DAYS SINCE 1/22/2020",size=30)
plt.ylabel("NO OF CASES",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(['Confirmed cases','Linear_Regression_prediction'])


# In[73]:


linear_pred[-10:]#normalize=True,fit_intercept=Tru


# #  BUILDING THE LOGISTIC MODEL TO PREDICT THE UPCOMING TOTAL  
# 
# # NO OF CONFIRMED CASES IN NEXT 10 DAYS

# In[74]:


from sklearn.linear_model import LogisticRegression
logi_model=LogisticRegression()
logi_model.fit(X_train_confirmed,y_train_confirmed)
test_logi_pred=logi_model.predict(X_test_confirmed)
logi_pred=logi_model.predict(future_forecast)
print("MAE:",mean_absolute_error(svm_test_pred,y_test_confirmed))
print("MSE:",mean_squared_error(svm_test_pred,y_test_confirmed))


# In[75]:


plt.plot(y_test_confirmed)
plt.plot(test_logi_pred)


# In[76]:


logi_pred


# In[77]:


plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,usa_cases)
plt.plot(future_forecast,logi_pred,linestyle="dashed",color='purple')
plt.title("NO OF CORONA CASE ",size=30)
plt.xlabel("DAYS SINCE 1/22/2020",size=30)
plt.ylabel("NO OF CASES",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.legend(['Confirmed cases','Logi_Regression_prediction'])


# #  "INTERESTING THING" the death should be increasesd from the 30 days from the first case bt till that time the situation is  under control .
# # But After  next 32 days the situtation was out of control and the confrimed cases start Increasing and it went on increasing the plot should be flatten bt the it went on increasing 

# In[83]:



print("==================================================================================")

print(list(logi_pred[-10:]))

print("==================================================================================")


# # HERE WE HAVE FULL COUNTRY DATA USING IT WE JUST PREDTICED WHEN THE CURE IS GOING TO BE FLATTEN OF ONLY USA
# 
# # ALMOST AFTER MORE 88 DAYS  USA IS IN DANGER ZONE

# In[94]:



#SO NOW WILL TAKE THE FULL DATA OF COUTRY AND SELECT THE USA AND PREDTIC THE FUTURE
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

train=pd.read_csv("D:\\RStudio\\mini COVID\\MODEL FOR COVID\\DATA SET UPDATED\\cleaned/covid-19-all.csv")


# In[97]:


train.tail(5)


# In[98]:


train.head(5)


# In[99]:


usa_df=train[train['Country/Region']=='US'].groupby('Date')['Confirmed','Deaths'].sum()


# In[100]:


usa_df


# In[101]:


usa_df['day_count']=list(range(1,len(usa_df)+1))#counting the date same full 115 as data is update now 115


# In[102]:


usa_df['day_count']


# In[105]:


xdata=usa_df.day_count
ydata=usa_df.Confirmed


# In[112]:


usa_df['rate'] = (usa_df.Confirmed - usa_df.Confirmed.shift(1))/usa_df.Confirmed


# In[113]:


usa_df['rate']


# In[114]:


usa_df['increase'] = (usa_df.Confirmed-usa_df.Confirmed.shift(1))


# In[115]:


usa_df['increase']


# In[116]:


plt.plot(xdata, ydata, 'o')
plt.title("USA")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.show()


# 
# 
# **Sigmoid function,
# 
# Here is a snap of how I learnt to fit Sigmoid Function - y = c/(1+np.exp(-a*(x-b))) and 3 coefficients [c, a, b]:
# 
#     c - the maximum value (eventual maximum infected people, the sigmoid scales to this value eventually)
#     a - the sigmoidal shape (how the infection progress. The smaller, the softer the sigmoidal shape is)
#     b - the point where sigmoid start to flatten from steepening (the midpoint of sigmoid, when the rate of increase start to slow down)
# 
# 

# In[117]:


us_df = train[train['Country/Region']=='US'].groupby('Date')['Confirmed','Deaths','Recovered'].sum()
us_df = us_df[us_df.Confirmed>=100]


# In[118]:


us_df


# In[119]:


from scipy.optimize import curve_fit
import pylab
from datetime import timedelta


# In[128]:


us_df['day_count'] = list(range(1,len(us_df)+1))
us_df['increase'] = (us_df.Confirmed-us_df.Confirmed.shift(1))
us_df['rate'] = (us_df.Confirmed-us_df.Confirmed.shift(1))/us_df.Confirmed
us_df['Active']=us_df['Confirmed']-us_df['Deaths']-us_df['Recovered']


# In[129]:


us_df['Active']


# In[130]:


us_df['day_count']


# In[131]:


us_df['increase']


# In[132]:


def sigmoid(x,c,a,b):
     y = c*1 / (1 + np.exp(-a*(x-b)))
     return y


# In[133]:


xdata = np.array(list(us_df.day_count)[::2])
ydata = np.array(list(us_df.Active)[::2])


# In[138]:


population=1.332*10**9
popt, pcov = curve_fit(sigmoid, xdata, ydata, method='dogbox',bounds=([0.,0., 0.],[population,6, 100.]))
print(popt)


# In[139]:


est_a = popt[0]
est_b = popt[1]
est_c = popt[2]


# In[140]:


est_a


# In[141]:


est_b


# In[142]:


est_c


# In[143]:


x = np.linspace(-1, us_df.day_count.max()+50, 50)
y = sigmoid(x,est_a,est_b,est_c)


# In[144]:


x


# In[145]:


y


# In[146]:


pylab.plot(xdata, ydata, 'o', label='data')
pylab.plot(x,y, label='fit',alpha = 0.6)
pylab.ylim(-0.05, est_a*1.05)
pylab.xlim(-0.05, est_c*2.05)
pylab.legend(loc='best')
plt.xlabel('days from day 1')
plt.ylabel('confirmed cases')
plt.title('USA')
pylab.show()


# In[147]:


print('model start date:',us_df[us_df.day_count==1].index[0])
print('model start infection:',int(us_df[us_df.day_count==1].Confirmed[0]))


# In[148]:


print('model fitted max Active at:',int(est_a))
print('model sigmoidal coefficient is:',round(est_b,3))
print('model curve stop steepening, start flattening by day:',int(est_c))


# In[149]:


print('model curve flattens by day:',int(est_c)*2)
display(us_df.head(3))
display(us_df.tail(3))


# In[79]:
