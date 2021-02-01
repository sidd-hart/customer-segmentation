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


processed_path = '/home/sid/mystuff/myprogs/flirt/projects/product_analytics/customer_segmentation/data/processed'


rfm_table.reset_index().to_csv?


rfm_table.reset_index().to_csv(
    os.path.join(processed_path, 'online_retail_rfm.csv'), 
    index=False)


rfm_csv = pd.read_csv(os.path.join(processed_path, 'online_retail_rfm.csv'))
rfm_csv.head()


rfm_train[rfm_train['CustomerID'] == rfm_table.index[0]]


# Check if the number difference of days from the purchase date in original record is same as shown in rfm table.
(current_date - rfm_train[rfm_train.CustomerID == rfm_table.index[0]].iloc[0].Purchase_Date).days == rfm_table.iloc[0,0]


## RFM Quantiles
quantiles = rfm_table.quantile(q=[0.25, 0.5, 0.75])
quantiles


quantiles = quantiles.to_dict()
quantiles


def RScore(x, p, d):
    '''
    Arguments (x = value, p = recency, monetary_value, frequency, d = quantiles dict) 
    '''
    if x <= d[p][0.25]:
        return 4
    elif x<= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1


def FMScore(x,p,d):
    '''
    Arguments (x = value, p = recency, monetary_value, frequency, k = quantiles dict)
    '''
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4


rfm_segment = rfm_table.copy()


rfm_segment['R_Quartile'] = rfm_segment['Recency'].apply(RScore, args=('Recency',quantiles,))
rfm_segment['F_Quartile'] = rfm_segment['Frequency'].apply(FMScore, args=('Frequency',quantiles,))
rfm_segment['M_Quartile'] = rfm_segment['Monetary'].apply(FMScore, args=('Monetary',quantiles,))


rfm_segment.head()


rfm_segment['RFMScore'] = rfm_segment.R_Quartile.map(str) \
                            + rfm_segment.F_Quartile.map(str) \
                            + rfm_segment.M_Quartile.map(str)
rfm_segment.head()


rfm_segment.reset_index(inplace=True)


rfm_segment.head()


# Create a dictionary for each segment to map them against each customer
segment_dict = {
    'Best Customers':'444',      # Highest frequency as well as monetary value with least recency
    'Loyal Customers':'344',     # High frequency as well as monetary value with good recency
    'Big Spenders':'334',        # High monetary value but good recency and frequency values
    'Almost Lost':'244',         # Customer's shopping less often now who used to shop a lot
    'Lost Customers':'144',      # Customer's shopped long ago who used to shop a lot.
    'Recent Customers':'443',    # Customer's who recently started shopping a lot but with less monetary value
    'Lost Cheap Customers':'122' # Customer's shopped long ago but with less frequency and monetary value
}


dict_segment = dict(zip(segment_dict.values(), segment_dict.keys()))


dict_segment


# Allocate segments to each customer as per the RFM score mapping
rfm_segment['Segment'] = rfm_segment.RFMScore.map(lambda x: dict_segment.get(x))


# Allocate all remaining customers to others segment category
rfm_segment.Segment.fillna('others', inplace=True)


rfm_segment.sample(10)


# Best Customers who's recency, frequency as well as monetary attribute is highest.
rfm_segment[rfm_segment.RFMScore=='444'].sort_values('Monetary', ascending=False).head()


# Biggest spenders
rfm_segment[rfm_segment.RFMScore=='334'].sort_values('Monetary', ascending=False).head()


# Almost Lost i.e. who's recency value is low
rfm_segment[rfm_segment.RFMScore=='244'].sort_values('Monetary', ascending=False).head()


# Lost customers that don't needs attention who's recency, frequency as well as monetary values are low
rfm_segment[rfm_segment.RFMScore=='122'].sort_values('Monetary', ascending=False).head()


# loyal customers who's purchase frequency is high
rfm_segment[rfm_segment.RFMScore=='344'].sort_values('Monetary', ascending=False).head()


# customers that you must retain are those whose monetary and frequency was high but recency reduced quite a lot recently
rfm_segment[rfm_segment.RFMScore=='244'].sort_values('Monetary', ascending=False).head()


# plot
fig, axes = plt.subplots(3, 1, figsize=(15, 15))
sns.distplot(rfm_table.Recency , color="dodgerblue", ax=axes[0], axlabel='Recency')
sns.distplot(rfm_table.Frequency , color="deeppink", ax=axes[1], axlabel='Frequency')
sns.distplot(rfm_table.Monetary , color="gold", ax=axes[2], axlabel='Monetary')
# plt.xlim(50,75);
plt.show();


rfm_table.describe().T


rfm_table_scaled = rfm_table.copy()


rfm_table_scaled.Monetary = rfm_table_scaled.Monetary + abs(rfm_table_scaled.Monetary.min()) + 1


rfm_table_scaled.Recency = rfm_table_scaled.Recency + abs(rfm_table_scaled.Recency.min()) + 1


rfm_table_scaled.describe().T


from sklearn.preprocessing import StandardScaler


log_df = np.log(rfm_table_scaled)


scaler = StandardScaler()
normal_df = scaler.fit_transform(log_df)
normal_df = pd.DataFrame(data=normal_df, index=rfm_table.index, columns=rfm_table.columns)


# plot again on the transformed RFM data
fig, axes = plt.subplots(3, 1, figsize=(15, 15))
sns.distplot(normal_df.Recency , color="dodgerblue", ax=axes[0], axlabel='Recency')
sns.distplot(normal_df.Frequency , color="deeppink", ax=axes[1], axlabel='Frequency')
sns.distplot(normal_df.Monetary , color="gold", ax=axes[2], axlabel='Monetary')
plt.show();


# find WCSS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(normal_df)
    wcss.append(kmeans.inertia_)


# plot elbow graph
plt.plot(range(1,11),wcss,marker='o');


from sklearn.metrics import silhouette_score
wcss_silhouette = []
for i in range(2,12):
    km = KMeans(n_clusters=i, random_state=0,init='k-means++').fit(normal_df)
    preds = km.predict(normal_df)    
    silhouette = silhouette_score(normal_df,preds)
    wcss_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))

plt.figure(figsize=(10,5))
plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,12)],y=wcss_silhouette,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in range(2,12)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()


kmeans = KMeans(n_clusters=4, random_state=1, init='k-means++')
kmeans.fit(normal_df)
cluster_labels = kmeans.labels_


kmeans


print(f"Shape of cluster label array is {cluster_labels.shape}")
print(f"Shape of RFM segment dataframe is {rfm_segment.shape}")


# Assign the clusters as column to each customer
Cluster_table = rfm_segment.assign(Cluster = cluster_labels)


Cluster_table.sample(10)


Cluster_table[Cluster_table.Cluster == 3].sample(5)


Cluster_table[Cluster_table.Cluster == 2].sample(5)


Cluster_table[Cluster_table.Cluster == 1].sample(5)


Cluster_table[Cluster_table.Cluster == 0].sample(5)


# Plotting two dimesional plots of each attributes respectively.
X = normal_df.iloc[:,0:3].values
count=X.shape[1]
for i in range(0,count):
    for j in range(i+1,count):
        plt.figure(figsize=(15,6));
        plt.scatter(X[cluster_labels == 0, i], X[cluster_labels == 0, j], s = 10, c = 'red', label = 'Cluster0')
        plt.scatter(X[cluster_labels == 1, i], X[cluster_labels == 1, j], s = 10, c = 'blue', label = 'Cluster1')
        plt.scatter(X[cluster_labels == 2, i], X[cluster_labels == 2, j], s = 10, c = 'green', label = 'Cluster2')
        plt.scatter(X[cluster_labels == 3, i], X[cluster_labels == 3, j], s = 10, c = 'cyan', label = 'Cluster3')
        plt.scatter(kmeans.cluster_centers_[:,i], kmeans.cluster_centers_[:,j], s = 50, c = 'yellow', label = 'Centroids')
        plt.xlabel(normal_df.columns[i])
        plt.ylabel(normal_df.columns[j])
        plt.legend()        
        plt.show();


# Assign Cluster values to each customer in normalized dataframe
normal_df = normal_df.assign(Cluster = cluster_labels)

# Melt normalized dataframe into long form to have all metric in same column
normal_melt = pd.melt(normal_df.reset_index(),
                      id_vars=['CustomerID','Cluster'],
                      value_vars=['Recency', 'Frequency', 'Monetary'],
                      var_name='Metric',
                      value_name='Value')
normal_melt.head()


# Assign Cluster labels to RFM table
rfm_table_cluster = rfm_table.assign(Cluster = cluster_labels)

# Average attributes for each cluster
cluster_avg = rfm_table_cluster.groupby(['Cluster']).mean() 

# Calculate the population average
population_avg = rfm_table.mean()

# Calculate relative importance of attributes by 
relative_imp = cluster_avg / population_avg - 1


plt.figure(figsize=(10, 5))
plt.title('Relative importance of attributes')
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show();






















































