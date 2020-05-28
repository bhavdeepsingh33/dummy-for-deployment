# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#print(os.getcwd())
#print(os.listdir('F:\ML Flask Projects\\templates'))
for dirname, _, filenames in os.walk('F:\ML Flask Projects\\templates'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% [code] {"scrolled":true}
pwd = os.getcwd()

df = pd.read_csv(pwd +"\Combined.csv")
for i in ['Mode', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month']:
    df[i] = pd.to_numeric(df[i], downcast='integer')
df.info()

df.head(10)


# ## Sklearn Implementation of DBSCAN:

# %% [markdown]
# ### Data Preprocessing:

# %% [code] {"scrolled":true}
data = df.copy()
data = data[:10000]
data=data.drop(['timestamp', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)

#df=pd.get_dummies(data)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
#num2 = scaler.fit_transform(data.drop(['timestamp'],axis=1))
num2 = scaler.fit_transform(data)
num2 = pd.DataFrame(num2, columns = data.columns)

# %% [code] {"scrolled":true}
from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(
 eps = .2, 
 metric='euclidean', 
 min_samples = 5,
 n_jobs = -1)
clusters = outlier_detection.fit_predict(num2)

# %% [code] {"scrolled":true}
clusters.shape

# %% [code] {"scrolled":true}
data['anomaly'] = pd.Series(clusters)
print(data.head())

# %% [code] {"scrolled":true}
data['anomaly'].unique()

# %% [code] {"scrolled":true}
X_anomaly = data[data['anomaly'] == -1]
X_normal = data[data['anomaly'] != -1]
print(X_anomaly.shape, X_normal.shape)


# %% [code] {"scrolled":true}



# %% [code] {"scrolled":true}
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE

# %% [code] {"scrolled":true}
print(data.columns)



# %% [code] {"scrolled":true}
anomaly_ind = data[data['anomaly']==-1].index
normal_ind = data[data['anomaly']!=-1].index


# %% [code] {"scrolled":true}
data.columns

# %% [code] {"scrolled":false}
import matplotlib.pyplot as plt
#import seaborn as sns

features = ['pCut::Motor_Torque',
       'pCut::CTRL_Position_controller::Lag_error',
       'pCut::CTRL_Position_controller::Actual_position',
       'pCut::CTRL_Position_controller::Actual_speed',
       'pSvolFilm::CTRL_Position_controller::Actual_position',
       'pSvolFilm::CTRL_Position_controller::Actual_speed',
       'pSvolFilm::CTRL_Position_controller::Lag_error', 'pSpintor::VAX_speed',
       'Mode']

for feature in features:
    plt.figure(figsize=(15,7))
    plt.plot(data[feature], color='blue', label = 'normal')
    plt.scatter(x=data.iloc[anomaly_ind].index, y=data.iloc[anomaly_ind][feature], color='red', label = 'anomalous')
    #plt.scatter(x=normal_pca[0], y=normal_pca[1], color='blue')
    plt.title(feature)
    plt.legend()


import pickle

pickle.dump(outlier_detection, open('DBSCAN_model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))


