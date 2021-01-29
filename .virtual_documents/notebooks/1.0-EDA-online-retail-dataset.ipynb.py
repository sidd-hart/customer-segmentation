import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.express as px

import os
import datetime as dt


plt.style.use('dark_background')
mpl.rcParams['figure.figsize'] = (12, 6)


data_path = '/home/sid/mystuff/myprogs/flirt/projects/product_analytics/customer_segmentation/data/raw'


dataset = os.path.join(data_path, 'online_retail.xlsx')


## need to install openpyxl first with get_ipython().getoutput("pip install openpyxl")

df = pd.read_excel(dataset, parse_dates=True, engine='openpyxl')
df.head()


df.shape


df.info()


df.describe().T


df['InvoiceDate'].describe()


## separate year and month from InvoiceDate
df['InvoiceYearMonth'] = df['InvoiceDate'].map(lambda d: 100 * d.year + d.month)


df.head()


df['Revenue'] = df['UnitPrice'] * df['Quantity']


## Monthly Revenue
df_revenue = df.groupby('InvoiceYearMonth')['Revenue'].sum().reset_index()


df_revenue


df['InvoiceYearMonth'] = pd.to_datetime(df['InvoiceYearMonth'], 
                                                format='get_ipython().run_line_magic("Y%m')", "")


df_revenue['InvoiceYearMonth'] = pd.to_datetime(df_revenue['InvoiceYearMonth'], 
                                                format='get_ipython().run_line_magic("Y%m')", "")


plt.plot(df_revenue['InvoiceYearMonth'], 
         df_revenue['Revenue'], marker='o')
plt.show()


df_revenue['MonthlyGrowth'] = df_revenue['Revenue'].pct_change()


df_revenue.head()


plt.plot(df_revenue['InvoiceYearMonth'], 
         df_revenue['MonthlyGrowth'], marker='o')
plt.show()


df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).astype(int)


df_uk = df.query("Country=='United Kingdom'").reset_index(drop=True)
df_uk.head()


## monthly acive users in uk
df_monthly_active = df_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()


df_monthly_active


plt.plot(df_monthly_active['InvoiceYearMonth'], 
         df_monthly_active['CustomerID'], marker='o')
plt.show()


df_monthly_sales = df_uk.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()


df_monthly_sales


df_monthly_sales['Quantity'].mean()


df_monthly_order_avg = df_uk.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()


df_monthly_order_avg


df_uk.info()


tx_min_purchase = df_uk.groupby('CustomerID').InvoiceDate.min().reset_index()


tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']


tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)


tx_min_purchase


tx_uk = pd.merge(df_uk, tx_min_purchase, on='CustomerID')


tx_uk['MinPurchaseYearMonth'] = pd.to_datetime(tx_uk['MinPurchaseYearMonth'], 
                                               format='get_ipython().run_line_magic("Y%m')", "")


tx_uk['UserType'] = 'New'
tx_uk.loc[tx_uk['InvoiceYearMonth']>tx_uk['MinPurchaseYearMonth'],'UserType'] = 'Existing'


tx_uk.UserType.value_counts()


tx_uk.head()


tx_user_type_revenue = tx_uk.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()


tx_user_type_revenue.query("InvoiceYearMonth get_ipython().getoutput("= 20101201 and InvoiceYearMonth != 20111201")")


tx_user_type_revenue = tx_user_type_revenue.query("InvoiceYearMonth get_ipython().getoutput("= 20101201 and InvoiceYearMonth != 20111201")")


tx_user_ratio = tx_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()/tx_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique() 
tx_user_ratio = tx_user_ratio.reset_index()
tx_user_ratio = tx_user_ratio.dropna()


tx_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()


tx_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()


tx_min_purchase['MinPurchaseYearMonth'] = pd.to_datetime(tx_min_purchase['MinPurchaseYearMonth'], 
                                               format='get_ipython().run_line_magic("Y%m')", "")


tx_min_purchase.head()


unq_month_year =  tx_min_purchase.MinPurchaseYearMonth.unique()


unq_month_year


def generate_signup_date(year_month):
    signup_date = [el for el in unq_month_year if year_month >= el]
    return np.random.choice(signup_date)


tx_min_purchase['SignupYearMonth'] = tx_min_purchase.apply(lambda row: generate_signup_date(row['MinPurchaseYearMonth']),axis=1)


tx_min_purchase['InstallYearMonth'] = tx_min_purchase.apply(lambda row: generate_signup_date(row['SignupYearMonth']),axis=1)


tx_min_purchase.head()


channels = ['organic','inorganic','referral']


tx_min_purchase['AcqChannel'] = tx_min_purchase.apply(lambda x: np.random.choice(channels),axis=1)


tx_activation = tx_min_purchase[tx_min_purchase['MinPurchaseYearMonth'] == tx_min_purchase['SignupYearMonth']].groupby('SignupYearMonth').CustomerID.count()/tx_min_purchase.groupby('SignupYearMonth').CustomerID.count()
tx_activation = tx_activation.reset_index()


tx_activation_ch = tx_min_purchase[tx_min_purchase['MinPurchaseYearMonth'] == tx_min_purchase['SignupYearMonth']].groupby(['SignupYearMonth','AcqChannel']).CustomerID.count()/tx_min_purchase.groupby(['SignupYearMonth','AcqChannel']).CustomerID.count()
tx_activation_ch = tx_activation_ch.reset_index()


tx_activation_ch.head(10)


df_monthly_active = tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()


tx_user_purchase = tx_uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().astype(int).reset_index()


tx_user_purchase


tx_user_purchase.Revenue.sum()


tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()


tx_retention.head()


months = tx_retention.columns[2:]


months


retention_array = []
for i in range(len(months)-1):
    retention_data = {}
    selected_month = months[i+1]
    prev_month = months[i]
    retention_data['InvoiceYearMonth'] = selected_month.strftime('get_ipython().run_line_magic("Y%m')", "")
    retention_data['TotalUserCount'] = tx_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = tx_retention[(tx_retention[selected_month]>0) & (tx_retention[prev_month]>0)][selected_month].sum()
    retention_array.append(retention_data)


tx_retention = pd.DataFrame(retention_array)
tx_retention.head(10)


tx_retention['RetentionRate'] = tx_retention['RetainedUserCount']/tx_retention['TotalUserCount']


tx_retention['InvoiceYearMonth'] = pd.to_datetime(tx_retention['InvoiceYearMonth'], 
                                               format='get_ipython().run_line_magic("Y%m')", "")
tx_retention.head()


tx_retention['ChurnRate'] =  1- tx_retention['RetentionRate']


tx_user_purchase.head()


tx_min_purchase.head()


tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()


tx_retention = pd.merge(tx_retention,tx_min_purchase[['CustomerID','MinPurchaseYearMonth']],on='CustomerID')


tx_retention.head()


tx_retention.columns


new_column_names = [ 'm_' + str(column) for column in tx_retention.columns[:-1]]
new_column_names.append('MinPurchaseYearMonth')


tx_retention.columns = new_column_names


tx_retention


months


retention_array = []

for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan
        
    total_user_count = tx_retention[tx_retention.MinPurchaseYearMonth == selected_month].MinPurchaseYearMonth.count()
    retention_data['TotalUserCount'] = total_user_count
    retention_data[selected_month] = 1 
    
    query = "MinPurchaseYearMonth == {}".format(selected_month)
    

    for next_month in next_months:
        new_query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(tx_retention.query(new_query)['m_' + str(next_month)].sum()/total_user_count,2)
    retention_array.append(retention_data)






























