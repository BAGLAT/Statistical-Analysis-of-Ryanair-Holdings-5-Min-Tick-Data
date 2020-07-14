#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pylab
import statsmodels.graphics.tsaplots as sgt
import seaborn as sns


# ### Reading the data (5min Intraday data for RYAAY Holdings)

# In[8]:


df = pd.read_csv("RYAAY.csv")


# In[9]:


df.head()


# ### Data Preprocessing

# In[10]:


df['DateTime'] = df['Date'] + ' ' + df['Time']


# In[11]:


df['DateTime'] = pd.to_datetime(df['DateTime'])


# In[12]:


df = df.set_index('DateTime')


# In[13]:


df.head()


# In[14]:


df = df.drop(['Date','Time'],axis=1)


# In[15]:


df.head()


# In[24]:


df1 = df.copy()


# In[25]:


#pd.options.display.max_rows = 50


# In[26]:


df1.isna().sum()


# In[437]:


#df1 = df1.dropna()


# In[438]:


#df1.isna()


# In[18]:


df1.describe()


# ### Feature Engineering

# In[27]:


df1['logReturn'] = np.log(df1.Close/df1.Close.shift(1))*100


# In[28]:


df1.head()


# In[31]:


df1['vol_sqd_return'] = round((df1['logReturn'])**2,6)


# In[32]:


df1.tail()


# In[351]:


df1['vol_sqd_return']


# ### Summarized Statistics

# In[91]:


df1.describe()


# ## Plotting Returns and Close Price

# In[92]:


plt.figure(figsize=(25,12))
plt.xlabel('Date')
plt.ylabel('LogReturn')
plt.plot('logReturn',data=df1,color='Crimson')


# In[94]:


plt.figure(figsize=(25,12))
plt.xlabel('Date')
plt.ylabel('Close')
plt.plot('Close',data=df1,color='Blue')


# In[96]:


plt.figure(figsize=(20, 12))
ax = sns.distplot(df1['logReturn'],
                  bins=300,
                  kde=True,
                  color='Red')
ax.set(xlabel='Normal Distribution', ylabel='Frequency')


# ## Synthetic Data Generation/Simulation Using Inverse CDF (Cumulative Distribution Function)

# In[97]:


df1['AbsLog'] = abs(df1['logReturn'])


# In[98]:


df1.head()


# In[99]:


## PDF
df1['pdf'] = df1['AbsLog']/(df1['AbsLog'].sum())


# In[101]:


df1.head()


# In[104]:


round(df1['pdf'].sum())


# In[105]:


df1['cdf'] = df1['pdf'].cumsum()


# In[106]:


df1.head()


# In[107]:


x = np.sort(df1['logReturn'])
y = df1['cdf']
plt.figure(figsize=(25,12))
plt.scatter(x=x, y=y);
plt.xlabel('LogReturn', fontsize=24)
plt.ylabel('Cumulative Probability Distribution', fontsize=24)


# ## Sampling (Interpolation)

# In[110]:


from scipy import interpolate


# In[116]:


x = np.array(df1['logReturn'])
y = np.array(df1['cdf'])


# In[112]:


Finv = interpolate.interp1d(y,x,bounds_error=False)


# In[113]:


xnew = np.random.rand(1000000)


# In[114]:


n = Finv(xnew)
gen_logR = pd.DataFrame(n,columns=['sampled_return'])


# In[115]:


gen_logR['sampled_return'].describe()


# In[118]:


df1['logReturn'].describe()


# In[124]:


from scipy import stats
result=pd.DataFrame(stats.norm.ppf(q=xnew,loc=df1['logReturn'].mean(),scale = df1['logReturn'].std()))


# In[125]:


result.describe()


# In[128]:


result.head()


# In[137]:


plt.figure(figsize=(20, 10))
ax = sns.distplot(df1['logReturn'],
                  bins=200,
                  kde=True,
                  color='blue')
ax = sns.distplot(gen_logR,
                  bins=200,
                  kde=True,
                  color='yellow')
ax.set(xlabel='logReturn', ylabel='Frequency')
plt.title('Simulated Data vs Given Distribution')


# ### Rolling Mean and Rolling SD

# ##### Close Prices

# In[152]:


df1.head()


# In[176]:


df1['rolling_mean_ST_20'] = df1.Close.rolling(window=20).mean()
df1['rolling_mean_LT_100'] = df1.Close.rolling(window=100).mean()
df1['rolling_SD_ST_20'] = df1.Close.rolling(window=20).std()
df1['rolling_SD_LT_100'] = df1.Close.rolling(window=100).std()


# In[177]:


df1.Close.plot(figsize=(20,10))
df1.rolling_mean_ST_20.plot(figsize=(20,10),color='black')
df1.rolling_mean_LT_100.plot(figsize=(20,10),color='green')
#df_1_day.rolling_mean_LT.plot(figsize=(20,5))
plt.xlabel("DateTime",size=15)
plt.ylabel("Close Prices",size=15)
plt.legend()


# ##### Returns Rolling Mean

# In[ ]:





# In[178]:


df1['rolling_mean_ST_20'] = df1.logReturn.rolling(window=20).mean()
df1['rolling_mean_LT_100'] = df1.logReturn.rolling(window=100).mean()
df1['rolling_SD_ST_20'] = df1.logReturn.rolling(window=20).std()
df1['rolling_SD_LT_100'] = df1.logReturn.rolling(window=100).std()


# In[179]:


df1.tail()


# In[180]:


df1.logReturn.plot(figsize=(20,10))
df1.vol_sqd_return.plot(figsize=(20,10))
df1.rolling_mean_ST_20.plot(figsize=(20,10),color='black')
df1.rolling_SD_ST_20.plot(figsize=(20,10),color='green')
#df_1_day.rolling_mean_LT.plot(figsize=(20,5))
plt.xlabel("DateTime",size=15)
plt.ylabel("Log Return",size=15)
plt.legend()


# In[181]:


#df_1_day.percent_logReturn.plot(figsize=(20,5),color='red')
df1.vol_sqd_return.plot(figsize=(20,5),color='red')
plt.xlabel("DateTime",size=15)
plt.ylabel("Squarred Returns/Volatility",size=15)
plt.title('Measure of Volatility',size=20)


# In[ ]:





# ## Stylized Facts

# ### 1. Distribution of returns is not normal
# ### QQ (Quantile Quantile Plot)

# In[182]:


scipy.stats.probplot(df1.logReturn,plot = pylab,fit=True)
pylab.show()


# #### A normal probability plot, or more specifically a quantile-quantile (Q-Q) plot, shows the distribution of the data against 
# #### the expected normal distribution If the data is non-normal, the points form a curve that deviates markedly from a straight line.

# In[183]:


from scipy import stats


# In[184]:


stats.skew(df1.logReturn[1:])


# In[185]:


plt.figure(figsize=(10, 8))
ax = sns.distplot(df1['logReturn'],
                  bins=100,
                  kde=True,
                  color='crimson')
ax.set(xlabel='logReturn', ylabel='Frequency')


# In[186]:


stats.kurtosis(df1.logReturn[1:])


# In[187]:


stats.jarque_bera(df1.logReturn[1:])


# In[188]:


import seaborn as sns
from matplotlib import style
sns.set(style="whitegrid")
plt.figure(figsize=(20, 12))
plt.xlabel('LogReturn')
style.use('ggplot')
sns.boxplot(x=df1['logReturn'])
plt.title('Boxplot to measure Outliers',size=15)


# ### 2. Checking for stationary using Dickey Fuller hypothesis Test

# In[189]:


import statsmodels.tsa.stattools as sts


# In[190]:


df1


# In[191]:


df1.isna().sum()


# In[192]:


df1 = df1.dropna()


# In[193]:


df1


# #### Check if close prices are not Stationary

# In[81]:


sts.adfuller(df1.Close)


# #### Null Hypothesis : Close Prices are not stationary
# #### T test statistic > Critical Value at 1% , 5% and 10% confidence level
# #### p value > 0.05 (5% level) 
# ##### Result : We can't reject null hypothesis--> Close Prices are not stationary

# #### Returns

# In[83]:


# Returns are Stationary
sts.adfuller(df1.logReturn)


# #### Null Hypothesis : Returns are not stationary
# #### T test statistic < Critical Value at 1% , 5% and 10% confidence level
# #### p value < 0.05 (5% level) - Significant
# #### Result : We reject null hypothesis--> Returns are stationary

# ## Autocorrelation ACF for prices

# In[84]:


### Prices are highly autocorrelated
sgt.plot_acf(df1.Close,lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.title("ACF LAG")


# ## Autocorrelation ACF for Returns

# In[85]:


### Returns are not much autocorrelated as shown by the significance level
sgt.plot_acf(df1.logReturn[1:],lags=40,zero=False)
plt.xlabel('Lags')
plt.ylabel('ACF for Normal Log Returns')
plt.title("ACF LAG")


# In[393]:


df1.head()


# In[ ]:




