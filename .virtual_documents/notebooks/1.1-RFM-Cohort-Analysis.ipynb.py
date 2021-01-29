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


## Import dataset
train = pd.read_excel(datapath, parse_dates=['InvoiceDate'], engine='openpyxl')


train.shape


train.head()


## check for duplicate items
train.duplicated().sum()


## remove duplicated items
train = train[~train.duplicated()]


## check for missing values
train.isnull().sum()


## create a list of unique InvoiceNo with null CustomerID records
Invoice_list = train[train['CustomerID'].isnull()]['InvoiceNo'].tolist()

Invoice_list


## Check number of records with these invoice number
len(train[train.InvoiceNo.isin(Invoice_list)])


## Removing records with null customer IDs
rfm_train = train[train['CustomerID'].notnull()].copy()


rfm_train.shape


rfm_train.info(memory_usage='deep')


rfm_train['CustomerID'] = rfm_train['CustomerID'].astype(int)


rfm_train.info(memory_usage='deep')


rfm_train.isnull().sum()


desc_df = rfm_train[~rfm_train['InvoiceNo'].str.contains('C', na=False)]


## Lets create total cost feature
desc_df['Total_cost'] = rfm_train['Quantity'] * rfm_train['UnitPrice']


desc_df.head()


print(f'Oldest date is - {desc_df.InvoiceDate.min()}\n')
print(f'Latest date is - {desc_df.InvoiceDate.max()}')


# Check the top ten countries in the dataset with highest transactions
desc_df.Country.value_counts(normalize=True).head(10).mul(100).round(1).astype(str) + 'get_ipython().run_line_magic("'", "")


# Count of transactions in different years
desc_df.InvoiceDate.dt.year.value_counts(sort=False).plot(kind='bar', rot=45);


# Count of transactions in different months within 2011 year.
desc_df[desc_df.InvoiceDate.dt.year==2011].InvoiceDate.dt.month.value_counts(sort=False).plot(kind='bar');


# Let's visualize the top grossing months
monthly_gross = desc_df[desc_df.InvoiceDate.dt.year==2011].groupby(desc_df.InvoiceDate.dt.month).Total_cost.sum()
plt.figure(figsize=(10,5))
sns.lineplot(y=monthly_gross.values,x=monthly_gross.index, marker='o');
plt.xticks(range(1,13))
plt.show();


desc_df.describe().T


# Boxplot to visualize the Quantity distribution
plt.figure(figsize=(16,4))
sns.boxplot(y='Quantity', data=desc_df, orient='h');


# Let's visualize the Unit price distribution
plt.figure(figsize=(16,4))
sns.boxplot(y='UnitPrice', data=desc_df, orient='h');


# Let's visualize some top products from the whole range.
top_products = desc_df['Description'].value_counts()[:20]


plt.figure(figsize=(10,6))
sns.set_context("paper", font_scale=1.5)
sns.barplot(y = top_products.index,
            x = top_products.values)
plt.title("Top selling products")
plt.show();


## create a copy of rfm_train df for cohort analysis
cohort = rfm_train.copy()


## function to parse dates
def get_month(x):
    return dt.datetime(x.year, x.month, 1)


## Create InvoiceMonth Column
cohort['InvoiceMonth'] = cohort['InvoiceDate'].apply(get_month)


cohort.head()


## Group by CustomerID and select InvoiceMonth Value
grouping = cohort.groupby('CustomerID')['InvoiceMonth']


## Assign a minimum InvoiceMonth value to the dataset
cohort['CohortMonth'] = grouping.transform('min')


def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month


invoice_year, invoice_month = get_date_int(cohort, 'InvoiceMonth')


cohort_year, cohort_month = get_date_int(cohort, 'CohortMonth')


years_diff = invoice_year - cohort_year


months_diff = invoice_month - cohort_month


cohort['CohortIndex'] = years_diff * 12 + months_diff + 1


cohort.head()


grouping = cohort.groupby(['CohortMonth', 'CohortIndex'])


cohort_data = grouping['CustomerID'].apply(pd.Series.nunique).reset_index()


cohort_data


cohort_counts = cohort_data.pivot(index='CohortMonth', 
                                  columns='CohortIndex', 
                                  values='CustomerID')


cohort_counts


cohort_sizes = cohort_counts.iloc[:, 0]
cohort_sizes


retention = cohort_counts.divide(cohort_sizes, axis=0) * 100


retention


month_list = ["Dec '10", "Jan '11", "Feb '11", "Mar '11", "Apr '11",\
              "May '11", "Jun '11", "Jul '11", "Aug '11", "Sep '11", \
              "Oct '11", "Nov '11", "Dec '11"]


px.imshow(
    img=retention,
    zmin=0.0,
    title='Retention by monthly cohorts',
    y = month_list
).show()


sns.heatmap(data=retention, 
            annot=True, 
            fmt='.1f', 
            linewidth=0.2, 
            yticklabels=month_list)
plt.show()


grouping = cohort.groupby(['CohortMonth', 'CohortIndex'])

cohort_data = grouping['UnitPrice'].mean()


cohort_data


cohort_data = cohort_data.reset_index()


average_price = cohort_data.pivot(index='CohortMonth', 
                                  columns='CohortIndex', 
                                  values='UnitPrice')


average_price.round(1)


average_price.index = average_price.index.date


sns.heatmap(data=average_price, 
            annot=True, 
            fmt='.1f', 
            linewidth=0.2, 
            yticklabels=month_list)
plt.title('Average spend by monthly cohorts')
plt.show()


cohort_data = grouping['Quantity'].mean()
cohort_data


cohort_data = cohort_data.reset_index()


cohort_data


# Create a pivot 
average_quantity = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='Quantity')


average_quantity.round(2)


sns.heatmap(data=average_quantity, 
            annot=True, 
            fmt='.1f', 
            linewidth=0.2, 
            yticklabels=month_list)
plt.title('Average quantity by monthly cohorts')
plt.show()


current_date = rfm_train['InvoiceDate'].max().date()


rfm_train['Purchase_Date'] = rfm_train.InvoiceDate.dt.date


recency = rfm_train.groupby('CustomerID')['Purchase_Date'].max().reset_index()


recency


recency['Current_Date'] = current_date


recency


## Compute the number of days since last purchase
recency['Recency'] = recency['Purchase_Date'].apply(lambda x: (current_date - x).days)


recency.head()


recency.drop(['Purchase_Date', 'Current_Date'], axis=1, inplace=True)


frequency = rfm_train.groupby('CustomerID')['InvoiceNo'].nunique().reset_index().rename(columns={'InvoiceNo': 'Frequency'})


frequency.head()


rfm_train['Total_cost'] = rfm_train['Quantity'] * rfm_train['UnitPrice']


monetary = rfm_train.groupby('CustomerID').Total_cost.sum().reset_index().rename(columns={'Total_cost': 'Monetary'})


monetary.head()


temp_ = recency.merge(frequency, on='CustomerID')
rfm_table = temp_.merge(monetary, on='CustomerID')


rfm_table.set_index('CustomerID', inplace=True)
rfm_table.head()













































