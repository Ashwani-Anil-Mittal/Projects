#!/usr/bin/env python
# coding: utf-8

# Assignment 3 Python Notebook
# 
# In this notebook, we will import local data from Yahoo Finance, Parse and clean the data and then create a program to calculate individual stock returns, standard deviation of returns, variance/co-variance matrix and a monte carlo simulation for 10,000 possible portfolios for 10 tickers.
# 
# In the first section, we will define a function to remove unnecessary columns, and add a symbol column to the end of each dataframe (one dataframe for each company). We will also add returns for each day into the dataframe using the same function

# In[1]:


#import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sci_opt

#define function to clean file data and convert to correct data types
def clean_data_calculate_returns(file_name,symbol):
    file_name = file_name[['Date','Adj Close']]
    file_name = file_name.rename(columns={'Date':'date','Adj Close':'price'})
    file_name = pd.DataFrame(file_name)
    file_name = file_name.astype({'price':float})
    file_name['return']=file_name['price'].pct_change(1)
    file_name['symbol']= symbol
    return file_name

#load all csv into pandas dataframe
#create list of all tickers
tickers = ['AAPL','BA','BAC','BBY','HD','HOG','IHG','RTX','SBUX','XOM']

#loop function to load data for each ticker
x = 0
tickers[0]
for i in tickers:
    tickers[x] = pd.read_csv('C:/Users/aniqa/OneDrive - The University of Texas at Dallas/Documents/Courses/Semester_1/Asset Pricing and Management/Assignment 3/'+tickers[x]+'.csv')
    x = x+1

#assign each dataframe created in the tickers list to a symbol
AAPL = tickers[0]
BA = tickers[1]
BAC = tickers[2]
BBY = tickers[3]
HD = tickers[4]
HOG = tickers[5]
IHG = tickers[6]
RTX = tickers[7]
SBUX = tickers[8]
XOM = tickers[9]

#apply data cleaning and return function to each company
AAPL = clean_data_calculate_returns(AAPL,'AAPL')
BA = clean_data_calculate_returns(BA,'BA')
BAC = clean_data_calculate_returns(BAC,'BAC')
BBY = clean_data_calculate_returns(BBY,'BBY')
HD = clean_data_calculate_returns(HD,'HD')
HOG = clean_data_calculate_returns(HOG,'HOG')
IHG = clean_data_calculate_returns(IHG,'IHG')
RTX = clean_data_calculate_returns(RTX,'RTX')
SBUX = clean_data_calculate_returns(SBUX,'SBUX')
XOM = clean_data_calculate_returns(XOM,'XOM')



# Now, we calculate the standard deviation of returns of each stock.

# In[2]:


#define variance and standard deviation function
def risk_calculator(company):
    company['variance'] = company['return'].var()
    company['risk'] = company['return'].std()
    return company

#function to swap columns to keep symbol at the end
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


#Calculate risk and variance and reorder the columns to be correct (RUN CELL ONLY ONCE to prevent changing order)
AAPL = risk_calculator(AAPL)
#AAPL = swap_columns(AAPL, 'risk', 'symbol')

BA = risk_calculator(BA)
#BA = swap_columns(BA,'risk','symbol')

BAC = risk_calculator(BAC)
#BAC = swap_columns(BAC,'risk','symbol')

BBY = risk_calculator(BBY)
#BBY = swap_columns(BBY,'risk','symbol')

HD = risk_calculator(HD)
#HD = swap_columns(HD,'risk','symbol')

HOG = risk_calculator(HOG)
#HOG = swap_columns(HOG,'risk','symbol')

IHG = risk_calculator(IHG)
#IHG = swap_columns(IHG,'risk','symbol')

RTX = risk_calculator(RTX)
#RTX = swap_columns(RTX,'risk','symbol')

SBUX = risk_calculator(SBUX)
#SBUX = swap_columns(SBUX,'risk','symbol')

XOM = risk_calculator(XOM)
#XOM = swap_columns(XOM,'risk','symbol')

XOM


# Now we can calculate the variance co-variance matrix of returns. But before we do that we have to transform the data once more to make it easier to calculate the values. We will create a new dataframe called returns, with date as the first column and the ticker symbols in subsequent columns. We will then use the covariance function to find the variance covariance matrix

# In[3]:


#to keep things simple we will start with a DataFrame containing only the dates

returns = pd.DataFrame(AAPL['date'])

#now we create a function to append returns from each of the previous dataframes to the returns dataframe
def return_dataframe(return_df, company, symbol):
    return_df[symbol] = company['return']
    return return_df

#create a loop to append returns from all companies into the returns dataframe
tickers = ['AAPL','BA','BAC','BBY','HD','HOG','IHG','RTX','SBUX','XOM']
companies = [AAPL,BA,BAC,BBY,HD,HOG,IHG,RTX,SBUX,XOM]


x=0
y=0
for (i,n) in zip(tickers, companies):
    return_dataframe(returns,companies[x],tickers[y])
    x=x+1
    y=y+1


# In[4]:


#use the .cov() method to create covariance matrix
covmatrix = returns.cov()
covmatrix


# In[6]:


mean_returns = returns.mean()
mean_returns


# In[13]:


#define number of iterations
num_iterations = 10000

#create array to hold simulation results
array_res = np.zeros((4+len(companies)-1, num_iterations))
np.shape(array_res)


# In[20]:


#Now we generate random weight (with a max weight of 10%)

for i in range(num_iterations):
    weights = np.array(np.random.uniform(0, 0.10, 10))
    weights /= np.sum(weights)
    portfolio_return = np.sum(mean_returns*weights)
    portfolio_stdev = np.sqrt(np.dot(weights.T,np.dot(covmatrix,weights)))
    array_res[0,i] = portfolio_return
    array_res[1,i] = portfolio_stdev
    array_res[2,i] = array_res[0,i]/array_res[1,i]
    for j in range(len(weights)):
        array_res[j+3,i] = weights[j]

simulation_results = pd.DataFrame(array_res.T, columns = ['return','stdev','sharpe',tickers[0],tickers[1],tickers[2],tickers[3],tickers[4],tickers[5],tickers[6],tickers[7],tickers[8],tickers[9]])
simulation_results


# In[35]:


weights = np.array(np.random.uniform(0, 1, 10))
weights /= np.sum(weights)
weights


# In[ ]:





# In[ ]:




