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


# %% [markdown]
# ### Sklearn Implementation of Elliptic Envelope:

# %% [code] {"scrolled":true}

data = df.copy()
data = data[:10000]
data=data.drop(['timestamp', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
#num2 = scaler.fit_transform(data.drop(['timestamp'],axis=1))
num2 = scaler.fit_transform(data)
num2 = pd.DataFrame(num2, columns = data.columns)


# %% [code] {"scrolled":true}
from sklearn.covariance import EllipticEnvelope
clf = EllipticEnvelope(contamination=.1,random_state=0)
clf.fit(num2)
ee_scores = pd.Series(clf.decision_function(num2)) 
ee_predict = clf.predict(num2)
ee_predict =  pd.Series(ee_predict).replace([-1,1],[1,0])

# %% [code] {"scrolled":true}
print(ee_scores)
print(ee_predict)

# %% [markdown]
# * ee_scores contains fitted densities.<br>
# * ee_predict contains labels, where -1 indicates an outlier and 1 does not. <br>
# * Labels are calculated based on clf.threshold_ and ee_scores.

# %% [code] {"scrolled":true}
anomaly_ind = ee_predict[ee_predict==1].index
anomaly_ind

# %% [code] {"scrolled":false}
import matplotlib.pyplot as plt

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
    cmap=np.array(['white','red'])
    plt.scatter(data.index, data[feature], c='blue',s=20)
    plt.scatter(anomaly_ind, data.iloc[anomaly_ind][feature],c='red')#,marker='x',s=100)
    plt.title(feature)
    plt.ylabel(feature)
    plt.show()

# %% [code] {"scrolled":true}
for feature in features:
    plt.figure(figsize=(12,8))
    plt.hist(data[feature], bins=50)
    plt.title(feature)
    plt.show()



    
import pickle

pickle.dump(clf, open('EllipticEnvelope_model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))
