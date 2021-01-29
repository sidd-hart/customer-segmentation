import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os


from sklearn.cluster import KMeans


data_folder = '/home/sid/mystuff/myprogs/flirt/projects/product_analytics/customer_segmentation/data/'


datapath = os.path.join(data_folder, 'processed/segmentation_scaled.csv')


df = pd.read_csv(datapath, index_col=0)
df.head()



























