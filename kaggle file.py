# -*- coding: utf-8 -*-


# %% [code] {"scrolled":true}
"""
import os
import glob
import pandas as pd
#os.chdir("/mydir")


files = [i for i in glob.glob('/kaggle/input/one-year-industrial-component-degradation/*.{}'.format('csv'))]
files


extension = 'csv'
all_filenames = [i for i in glob.glob('/kaggle/input/one-year-industrial-component-degradation/*[mode1].{}'.format(extension))] + \
                [i for i in glob.glob('/kaggle/input/one-year-industrial-component-degradation/oneyeardata/*[mode1].{}'.format(extension))]
#print(all_filenames)

#combine all files in the list
df = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
df.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')

"""

# %% [code] {"scrolled":false}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"scrolled":false}
"""
filenames = os.listdir('/kaggle/input/one-year-industrial-component-degradation/')
filenames = [i.strip(".csv") for i in filenames]
filenames.sort()
filenames.remove('oneyeardata')

parsed_filenames = []
for name in filenames:
    temp = name.split("T")
    month, date = temp[0].split("-")
    rhs = temp[1].split("_")
    hours, minutes, seconds = rhs[0][:2], rhs[0][2:4], rhs[0][4:]
    sample_no = rhs[1]
    mode = rhs[2][-1]
    # Now we have Month, Date, Hours, Minutes, Seconds, Sample Number, Mode 
    parsed_filenames.append([month, date, hours, minutes, seconds, sample_no, mode])
    
parsed_filenames = pd.DataFrame(parsed_filenames, columns=["Month", "Date", "Hours", "Minutes", "Seconds", "Sample Number", "Mode"])

for i in parsed_filenames.columns:
    parsed_filenames[i] = pd.to_numeric(parsed_filenames[i], errors='coerce')



path = '/kaggle/input/one-year-industrial-component-degradation/'
df = pd.DataFrame()
#f = pd.read_csv(path+filenames[0]+".csv")
#f = f.join(parsed_filenames[0:1], how='left')
#f = f.fillna(method='ffill')
#f
for ind, file in enumerate(filenames):
    file_content = pd.read_csv(path+file+".csv")
    file_content = file_content.join(parsed_filenames[ind:ind+1], how='left')
    file_content.fillna(method='ffill', inplace=True)
    
    if df.empty:
        df = file_content
        df.fillna(method='ffill', inplace=True)
    else:
        df = df.append(file_content, ignore_index=True)
        df.fillna(method='ffill', inplace=True)

        
for i in ['Mode', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month']:
    df[i] = pd.to_numeric(df[i], downcast='integer')
df.info()

"""

# %% [code] {"scrolled":true}
"""
if not os.path.exists('/kaggle/working/compiled_df'):
    os.makedirs('/kaggle/working/compiled_df')
#Saves dataframe to a csv file, removes a index
df.to_csv('/kaggle/working/compiled_df/Combined.csv', index=False)

"""

# %% [code] {"scrolled":true}
pwd = os.getcwd()

df = pd.read_csv(pwd +"\Combined.csv")
for i in ['Mode', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month']:
    df[i] = pd.to_numeric(df[i], downcast='integer')
df.info()

# %% [code] {"scrolled":true}
df.head(10)

# %% [markdown]
# ## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# 
# This is a clustering algorithm (an alternative to K-Means) that clusters points together and identifies any points not belonging to a cluster as outliers. It’s like K-means, except the number of clusters does not need to be specified in advance.
# 
# ### The method, step-by-step:
# 1. Randomly select a point not already assigned to a cluster or designated as an outlier. Determine if it’s a core point by seeing if there are at least min_samples points around it within epsilon distance.
# 2. Create a cluster of this core point and all points within epsilon distance of it (all directly reachable points).
# 3. Find all points that are within epsilon distance of each point in the cluster and add them to the cluster. Find all points that are within epsilon distance of all newly added points and add these to the cluster. Rinse and repeat. (i.e. perform “neighborhood jumps” to find all density-reachable points and add them to the cluster).
# 
# ### Lingo underlying the above:
# 1. Any point that has at least min_samples points within epsilon distance of it will form a cluster. This point is called a core point. The core point will itself count towards the min_samples requirement.
# 2. Any point within epsilon distance of a core point, but does not have min_samples points that are within epsilon distance of itself is called a borderline point and does not form its own cluster.
# 3. A border point that is within epsilon distance of multiple core points (multiple epsilon balls) will arbitrarily end up in just one of these resultant clusters.
# 4. Any point that is randomly selected that is not found to be a core point or a borderline point is called a noise point or outlier and is not assigned to any cluster. Thus, it does not contain at least min_samples points that are within epsilon distance from it or is not within epsilon distance of a core point.
# 5. The epsilon-neighborhood of point p is all points within epsilon distance of p, which are said to be directly reachable from p.
# 6. A point contained in the neighborhood of a point directly reachable from p is not necessarily directly reachable from p, but is density-reachable.
# 7. Any point that can be reached by jumping from neighborhood to neighborhood from the original core point is density-reachable.
# 
# ### Implementation Considerations:
# 1. You may need to standardize / scale / normalize your data first.
# 2. Be mindful of data type and the distance measure. I’ve read that the gower distance metric can be used for mixed data types. I’ve implemented Euclidean, here, which needs continuous variables, so I removed gender.
# 3. You will want to optimize epsilon and min_samples.

# %% [markdown]
# ## Sklearn Implementation of DBSCAN:

# %% [markdown]
# ### Data Preprocessing:

# %% [code] {"scrolled":true}
data = df.copy()
data = data[:10000]
data=data.drop(['Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)

#df=pd.get_dummies(data)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
num2 = scaler.fit_transform(data.drop(['timestamp'],axis=1))
num2 = pd.DataFrame(num2, columns = data.drop(['timestamp'],axis=1).columns)

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
data.head()

# %% [code] {"scrolled":true}
data['anomaly'].unique()

# %% [code] {"scrolled":true}
X_anomaly = data[data['anomaly'] == -1]
X_normal = data[data['anomaly'] != -1]
print(X_anomaly.shape, X_normal.shape)


# %% [code] {"scrolled":true}


# %% [markdown]
# DBSCAN will output an array of -1’s and 0’s, where -1 indicates an outlier. Below, I visualize outputted outliers in red by plotting two variables.

# %% [code] {"scrolled":true}
#from matplotlib import cm
#cmap = cm.get_cmap('Set1’)
#data.plot.scatter(x='Spend_Score',y='Income', c=clusters, cmap=cmap, colorbar = False)

# %% [code] {"scrolled":true}
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# %% [code] {"scrolled":true}
data.columns

# %% [code] {"scrolled":true}
"""
cols = ['pCut::Motor_Torque',
       'pCut::CTRL_Position_controller::Lag_error',
       'pCut::CTRL_Position_controller::Actual_position',
       'pCut::CTRL_Position_controller::Actual_speed',
       'pSvolFilm::CTRL_Position_controller::Actual_position',
       'pSvolFilm::CTRL_Position_controller::Actual_speed',
       'pSvolFilm::CTRL_Position_controller::Lag_error', 'pSpintor::VAX_speed',
       'Mode']

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[cols].values)

#data['pca-one'] = data_pca[:,0]
#data['pca-two'] = data_pca[:,1] 

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

"""

# %% [code] {"scrolled":true}
anomaly_ind = data[data['anomaly']==-1].index
normal_ind = data[data['anomaly']!=-1].index

# %% [code] {"scrolled":true}
#anomaly_pca = pd.DataFrame(data_pca[anomaly_ind])
#normal_pca = pd.DataFrame(data_pca[normal_ind])
#anomaly_pca

# %% [code] {"scrolled":true}
data.columns

# %% [code] {"scrolled":false}
import matplotlib.pyplot as plt
import seaborn as sns

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
    

# %% [markdown]
# # Isolation Forests
# 
# ### For each observation, do the following:
# 1. Randomly select a feature and randomly select a value for that feature within its range.
# 2. If the observation’s feature value falls above (below) the selected value, then this value becomes the new min (max) of that feature’s range.
# 3. Check if at least one other observation has values in the range of each feature in the dataset, where some ranges were altered via step 2. If no, then the observation is isolated.
# 4. Repeat steps 1–3 until the observation is isolated. The number of times you had to go through these steps is the isolation number. The lower the number, the more anomalous the observation is.

# %% [markdown]
# ## Sklearn Implementation of Isolation Forests:

# %% [code] {"scrolled":true}
data = df.copy()
data = data[:10000]
data=data.drop(['timestamp', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)


# %% [code] {"scrolled":false}
from sklearn.ensemble import IsolationForest
rs=np.random.RandomState(0)
clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1) 
clf.fit(data)
if_scores = clf.decision_function(data)
if_anomalies=clf.predict(data)
#print(if_anomalies)
if_anomalies=pd.Series(if_anomalies).replace([-1,1],[1,0])
#print(if_anomalies)
#if_anomalies=num[if_anomalies==1];

# %% [markdown]
# Below, I plot a histogram of if_scores values. Lower values indicate observations that are more anomalous.

# %% [code] {"scrolled":true}
plt.figure(figsize=(12,8))
plt.hist(if_scores);
plt.title('Histogram of Avg Anomaly Scores: Lower => More Anomalous');

# %% [markdown]
# Below, I plot observations identified as anomalies. These observations have if_scores values below the clf.threshold_ value.

# %% [code] {"scrolled":true}
anomaly_ind = if_anomalies[if_anomalies==1].index

# %% [code] {"scrolled":false}
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
    #cmap=np.array(['white','red'])
    plt.scatter(data.index,data[feature],c='green', label = 'normal')
    plt.scatter(anomaly_ind,data.iloc[anomaly_ind][feature],c='red', label='anomaly')
    plt.ylabel(feature)
    plt.title(feature)
    plt.legend()
    
    

    


# %% [markdown]
# # One-Class Support Vector Machines
# 
# #### The nu hyperparameter seems to be like the contamination hyperparameter in other methods. It sets the % of observations the algorithm will identify as outliers.

# %% [markdown]
# ### Sklearn Implementation of One-Class SVM:

# %% [code] {"scrolled":false}
data = df.copy()
#data = data[:10000]
data=data.drop(['timestamp', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)

# %% [code] {"scrolled":true}
from sklearn import svm
clf=svm.OneClassSVM(nu=.1,kernel='rbf', gamma='auto')
clf.fit(data)
y_pred=clf.predict(data)

# %% [code] {"scrolled":false}
y_pred = pd.Series(y_pred).replace([-1,1],[1,0])

# %% [markdown]
# Below, I plot observations identified as anomalies:

# %% [code] {"scrolled":false}
anomaly_ind = y_pred[y_pred==1].index

# %% [code] {"scrolled":true}
anomaly_ind

# %% [code] {"scrolled":true}
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
    #cmap=np.array(['white','red'])
    plt.scatter(data.index,data[feature],c='green', label = 'normal')
    plt.scatter(anomaly_ind,data.iloc[anomaly_ind][feature],c='red', label='anomaly')
    plt.ylabel(feature)
    plt.title(feature)
    plt.legend()

# %% [code] {"scrolled":true}


# %% [markdown]
# # Local Outlier Factor
# 
# LOF uses density-based outlier detection to identify local outliers, points that are outliers with respect to their local neighborhood, rather than with respect to the global data distribution. The higher the LOF value for an observation, the more anomalous the observation.
# 
# This is useful because not all methods will not identify a point that’s an outlier relative to a nearby cluster of points (a local outlier) if that whole region is not an outlying region in the global space of data points.
# 
# A point is labeled as an outlier if the density around that point is significantly different from the density around its neighbors.
# 
# In the below feature space, LOF is able to identify P1 and P2 as outliers, which are local outliers to Cluster 2 (in addition to P3).

# %% [code] {"scrolled":true}
data = df.copy()
data = data[:10000]
data=data.drop(['timestamp', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)

# %% [code] {"scrolled":false}
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

features = ['pCut::Motor_Torque',
       'pCut::CTRL_Position_controller::Lag_error',
       'pCut::CTRL_Position_controller::Actual_position',
       'pCut::CTRL_Position_controller::Actual_speed',
       'pSvolFilm::CTRL_Position_controller::Actual_position',
       'pSvolFilm::CTRL_Position_controller::Actual_speed',
       'pSvolFilm::CTRL_Position_controller::Lag_error', 'pSpintor::VAX_speed',
       'Mode']

def LOF_plot(k):
    var1,var2=1,2
    clf = LocalOutlierFactor(n_neighbors=k, contamination=.1)
    y_pred = clf.fit_predict(data)
    LOF_Scores = clf.negative_outlier_factor_
    
    y_pred = pd.Series(y_pred).replace([-1,1],[1,0])
    anomaly_ind = y_pred[y_pred==1].index
    
    for feature in features:
        plt.figure(figsize=(19,7))
        plt.title("Local Outlier Factor (LOF), K={}".format(k))
        plt.scatter(data.index,data[feature], color='k', s=3., label='Data points')
        radius = (LOF_Scores.max() - LOF_Scores) / (LOF_Scores.max() - LOF_Scores.min())
        plt.scatter(anomaly_ind,data.iloc[anomaly_ind][feature], s=1000 * radius, edgecolors='r',
        facecolors='none', label='Outlier scores')
        plt.axis('tight')
        plt.ylabel("{}".format(feature))
        plt.xlabel("{}".format(feature))
        legend = plt.legend(loc='upper left')
        legend.legendHandles[0]._sizes = [10]
        legend.legendHandles[1]._sizes = [20]
        plt.show();
    
    
#LOF_plot(5)
#LOF_plot(30)
LOF_plot(70)

# %% [markdown]
# ### Sklearn Implementation of Local Outlier Factor:

# %% [code] {"scrolled":true}
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=30, contamination=.1)
y_pred = clf.fit_predict(data)
LOF_Scores = clf.negative_outlier_factor_
LOF_pred=pd.Series(y_pred).replace([-1,1],[1,0])
LOF_anomalies=data[LOF_pred==1]

# %% [code] {"scrolled":true}
LOF_anomalies

# %% [code] {"scrolled":true}
LOF_anomalies.index

# %% [code] {"scrolled":false}
features

# %% [code] {"scrolled":true}
for feature in features:
    plt.figure(figsize=(15,7))
    cmap=np.array(['white','red'])
    plt.scatter(data.index,data[feature],c='blue',s=20)
    plt.scatter(LOF_anomalies.index,data.iloc[LOF_anomalies.index][feature],c='red')
     #,marker=’x’,s=100)
    plt.title(feature)
    plt.ylabel(feature)
    plt.show()
    #plt.scatter(data.index,data[feature], color='k', s=3., label='Data points')
    #radius = (LOF_Scores.max() - LOF_Scores) / (LOF_Scores.max() - LOF_Scores.min())
    #plt.scatter(anomaly_ind,data.iloc[anomaly_ind][feature], s=1000 * radius, edgecolors='r',

# %% [markdown]
# # Elliptic Envelope
# 
# The Elliptic Envelope method fits a multivariate gaussian distribution to the dataset. Use the contamination hyperparameter to specify the percentage of observations the algorithm will assign as outliers.

# %% [markdown]
# ### Sklearn Implementation of Elliptic Envelope:

# %% [code] {"scrolled":true}
data = df.copy()
data = data[:10000]
data=data.drop(['timestamp', 'Sample Number', 'Seconds', 'Minutes', 'Hours', 'Date', 'Month'],axis=1)

# %% [code] {"scrolled":true}
from sklearn.covariance import EllipticEnvelope
clf = EllipticEnvelope(contamination=.1,random_state=0)
clf.fit(data)
ee_scores = pd.Series(clf.decision_function(data)) 
ee_predict = clf.predict(data)
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

# %% [code] {"scrolled":true}
ee_predict

# %% [code] {"scrolled":true}


# %% [code] {"scrolled":true}











"""
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
    
"""