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


bgf = BetaGeoFitter(penalizer_coef=0.0)


bgf.fit(data['frequency'], data['recency'], data['T'])
print(bgf)


from lifetimes.plotting import plot_frequency_recency_matrix


fig = plt.figure(figsize=(12, 8))
plot_frequency_recency_matrix(bgf);


from lifetimes.plotting import plot_probability_alive_matrix


plot_probability_alive_matrix(bgf);


t = 1
data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t, 
    data['frequency'], 
    data['recency'],
    data['T']
)


data.sample(5)


data.sort_values(by='predicted_purchases').tail(5)


from lifetimes.plotting import plot_period_transactions


plot_period_transactions(bgf);


from lifetimes.utils import calibration_and_holdout_data


summary_cal_holdout = calibration_and_holdout_data(
    df, 
    'CustomerID', 
    'InvoiceDate', 
    calibration_period_end='2011-06-08',
    observation_period_end='2011-12-9'
)


display(summary_cal_holdout.head())


from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases


bgf.fit(summary_cal_holdout['frequency_cal'], 
        summary_cal_holdout['recency_cal'], summary_cal_holdout['T_cal'])


plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout);


t = 10
individual = data.loc[12347]


bgf.predict(t, individual['frequency'], individual['recency'], individual['T'])


from lifetimes.plotting import plot_history_alive


id = 14606
days_since_birth = 365
sp_trans = df.loc[df['CustomerID'] == id]


with plt.style.context('seaborn'):
    plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate');


id = 14720
days_since_birth = 365
sp_trans = df.loc[df['CustomerID'] == id]
with plt.style.context('seaborn'):
    plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate');



















































