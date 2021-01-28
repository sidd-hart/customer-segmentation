import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_folder = '/home/sid/mystuff/myprogs/flirt/projects/product_analytics/customer_segmentation/data/'


df = pd.read_csv(data_folder + 'raw/segmentation_data.csv')


df.head()


df.shape


df.info(memory_usage='deep')


for column in df.columns:
    print('********', column)
    print(df[column].nunique())


df.describe().T


import scipy.stats as stats


fig, axes = plt.subplots(nrows=len(df.columns[1:]), figsize=(30, 50))

for ax, col in zip(axes, df.columns[1:]):
    sns.distplot(df[col], ax=ax)
   # plt.title(col)
    plt.tight_layout()
    
plt.show()


stats.probplot(df['Age'], dist='norm', plot=plt)
plt.show()


stats.probplot(df['Income'], dist='norm', plot=plt)
plt.show()


plt.figure(figsize=(3, 6))
sns.boxplot(y=df['Age']);


plt.figure(figsize=(3, 6))
sns.boxplot(y=df['Income']);


def boundaries(df, column, dist):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    
    lower = df[column].quantile(0.25) - iqr * dist
    upper = df[column].quantile(0.75) + iqr * dist
    
    return upper, lower


upper_age, lower_age = boundaries(df, 'Age', 1.5)
upper_age, lower_age


upper_income, lower_income = boundaries(df, 'Income', 1.5)
upper_income, lower_income


df.max() - df.min()


corr = df.corr()


sns.heatmap(corr, annot=True);


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


df_scaled = scaler.fit_transform(df)


df_scaled


df_scaled = pd.DataFrame(df_scaled, columns=df.columns.values)


df_scaled.head()


df_scaled.to_csv(data_folder + 'processed/segmentation_scaled.csv')









