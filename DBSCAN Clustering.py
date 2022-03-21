#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    X, y = make_blobs(n_samples = numSamples, centers = centroidLocation,
                      cluster_std = clusterDeviation)
    
    X = StandardScaler().fit_transform(X)
    return X,y


# In[5]:


X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)


# In[6]:


epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
labels


# In[7]:



core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask


# In[8]:



n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_


# In[9]:



unique_labels = set(labels)
unique_labels


# In[10]:



colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))


# In[11]:



for k, col in zip(unique_labels, colors):
    if k == -1:
        
        col = 'k'

    class_member_mask = (labels == k)


    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)


# In[12]:


import csv
import pandas as pd
import numpy as np

filename='weather-stations20140101-20141231.csv'

pdf = pd.read_csv(filename)
pdf.head(5)


# In[13]:


pdf = pdf[pd.notnull(pdf["Tm"])]
pdf = pdf.reset_index(drop=True)
pdf.head(5)


# In[14]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (14,10)

llon=-140
ulon=-50
llat=40
ulat=65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()

my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()


xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist()
pdf['ym'] =ys.tolist()


for index,row in pdf.iterrows():

    my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)

plt.show()


# In[15]:


from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)


db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 



pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)


# In[16]:


set(labels)


# In[17]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map.drawcoastlines()
my_map.drawcountries()

my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()


colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))




for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))


# In[18]:


from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)


db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 



pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)


# In[19]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (14,10)

my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, 
            urcrnrlon=ulon, urcrnrlat=ulat)
my_map.drawcoastlines()
my_map.drawcountries()

my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()


colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))




for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))


# In[ ]:




