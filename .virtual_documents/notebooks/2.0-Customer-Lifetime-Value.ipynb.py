import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import datetime as dt
import os
import time


plt.style.use('dark_background')
mpl.rcParams['figure.figsize'] = (12, 6)


raw_folder = '/home/sid/mystuff/myprogs/flirt/projects/product_analytics/customer_segmentation/data/raw'
datapath = os.path.join(raw_folder, 'online_retail.xlsx')


data = pd.read_excel(datapath, parse_dates=['InvoiceDate'], engine='openpyxl')


data.head()


data.shape


# Feature selection
features = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']


data_clv = data[features].copy()


data_clv['TotalSales'] = data_clv['Quantity'].multiply(data_clv['UnitPrice'])


data_clv.shape


data_clv.head()


data_clv.describe().T


## drop negative values in Quantity & UnitPrice
data_clv = data_clv[data_clv['TotalSales'] > 0]
data_clv.describe()


## Check for missing value
pd.DataFrame(zip(data_clv.isnull().sum(), data_clv.isnull().sum()/len(data_clv)), 
             columns=['Count', 'Proportion'], index=data_clv.columns)


## dropping the null CustomerID values
data_clv = data_clv[pd.notnull(data_clv['CustomerID'])]


data_clv.info()





## Check for missing value
pd.DataFrame(zip(data_clv.isnull().sum(), data_clv.isnull().sum()/len(data_clv)), 
             columns=['Count', 'Proportion'], index=data_clv.columns)


# Printing the details of the dataset
maxdate = data_clv['InvoiceDate'].dt.date.max()
mindate = data_clv['InvoiceDate'].dt.date.min()
unique_cust = data_clv['CustomerID'].nunique()
tot_quantity = data_clv['Quantity'].sum()
tot_sales = data_clv['TotalSales'].sum()

print(f"The Time range of transactions is: {mindate} to {maxdate}")
print(f"Total number of unique customers: {unique_cust}")
print(f"Total Quantity Sold: {tot_quantity}")
print(f"Total Sales for the period: {tot_sales}")


customer = data_clv.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (x.max() - x.min()).days,
    'InvoiceNo': lambda x: len(x),
    'TotalSales': lambda x: sum(x)
})

customer.columns = ['Age', 'Frequency', 'TotalSales']
customer.head()


## Calculating the necessary variables for CLV calculation
Average_sales = round(np.mean(customer['TotalSales']), 2)
Average_sales


Purchase_freq = round(np.mean(customer['Frequency']), 2)
Purchase_freq


Retention_rate = round(customer[customer['Frequency'] > 1].shape[0] / customer.shape[0], 2)

churn = round(1 - Retention_rate, 2)


Retention_rate, churn


## calculating the CLV
## Assumin profit margin of 5%

Profit_margin = 0.05

CLV = round(((Average_sales * Purchase_freq / churn)) * Profit_margin)


CLV


# Transforming the data to customer level for the analysis
customer = data_clv.groupby('CustomerID').agg({'InvoiceDate':lambda x: x.min().month, 
                                                   'InvoiceNo': lambda x: len(x),
                                                  'TotalSales': lambda x: np.sum(x)})

customer.columns = ['Start_Month', 'Frequency', 'TotalSales']
customer.head()


# Calculating CLV for each cohort
months = ['Jan', 'Feb', 'March', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Monthly_CLV = []


for i in range(1, 13):
    customer_m = customer[customer['Start_Month']==i]
    
    Average_sales = round(np.mean(customer_m['TotalSales']),2)
    
    Purchase_freq = round(np.mean(customer_m['Frequency']), 2)
    
    Retention_rate = customer_m[customer_m['Frequency']>1].shape[0]/customer_m.shape[0]
    churn = round(1 - Retention_rate, 2)
    
    CLV = round(((Average_sales * Purchase_freq/churn)) * Profit_margin, 2)
    
    Monthly_CLV.append(CLV)


monthly_clv = pd.DataFrame(zip(months, Monthly_CLV), columns=['Months', 'CLV'])
display(monthly_clv.style.background_gradient())


















