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


import lifetimes.plotting as lp
import lifetimes.utils as lu
import lifetimes.fitters as lf


plt.style.use('dark_background')
mpl.rcParams['figure.figsize'] = (12, 6)


raw_folder = '/home/sid/mystuff/myprogs/flirt/projects/product_analytics/customer_segmentation/data/raw'
datapath = os.path.join(raw_folder, 'online_retail.xlsx')


df = pd.read_excel(datapath, parse_dates=['InvoiceDate'], engine='openpyxl')


df.head()


df['InvoiceDate'] = df['InvoiceDate'].dt.date


df.sample(5)


df = df[pd.notnull(df['CustomerID'])]
df = df[(df['Quantity'] > 0)]
df.sample(5)


df.shape


df['Sales'] = df['Quantity'] * df['UnitPrice'] ## similar to Revenue


cols_of_interest = ['CustomerID', 'InvoiceDate', 'Sales']

df = df[cols_of_interest]


df['CustomerID'].nunique()


df.sample(5)


data = lu.summary_data_from_transaction_data(df, 
                                             'CustomerID', 
                                             'InvoiceDate', monetary_value_col='Sales', 
                                             observation_period_end='2011-12-9'                                           
                                            )


data.head(10)


data.shape


data['frequency'].plot(kind='hist', bins=50);


print(data['frequency'].describe())
print(sum(data['frequency'] == 0)/float(len(data)))


from lifetimes import BetaGeoFitter


bgf = BetaGeoFitter














































































