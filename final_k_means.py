#!/usr/bin/env python
# coding: utf-8

# In[195]:


import time
import random
import numpy as np
import pandas as pd
from sklearn import cluster, datasets, metrics
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from pyspark.sql import SparkSession
from pyspark.sql import Row


# In[196]:


spark = SparkSession                 .builder                 .appName("RDD_and_DataFrame")                 .config("spark.some.config.option", "some-value")                 .getOrCreate()
sc = spark.sparkContext


# # Data Preprocessing

# ## Wine Dataset

# In[197]:


'''
line = sc.textFile("./dataset/wine.txt")
data_x = spark.createDataFrame(line.map(lambda r : r.split(",")).collect()).toPandas()
converge_dist = 0.05
K = 5
max_iter = 30
'''


# In[198]:



line = sc.textFile("./dataset/c20d6n200000.txt")
data_x = spark.createDataFrame(line.map(lambda r : r.split(",")).collect()).toPandas()
converge_dist = 0.05
K = 20
max_iter = 30


# In[199]:


def cal_distance(p,cent):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(cent)):
        tempDist = np.sum((p - cent[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
            
    return bestIndex


# ## Here comes our k-means

# In[200]:


start = time.time()
#initialize tables
data = spark.createDataFrame(data_x).rdd.map(lambda r : np.array([float(x) for x in r])).cache()
cent = data.takeSample(False,K,1)
table= np.zeros(K)   
                
#first iteration    
iterl = 0
curr_sse = 1.0
        
#update centroid & data clustering
while iterl<max_iter and curr_sse > converge_dist :
    mapping_data = data.map(lambda point : (cal_distance(point,cent) , (point,1)))
    table = mapping_data.reduceByKey(lambda p1,p2 : (p1[0]+p2[0] , p1[1]+p2[1]))
            
    result = table.map(lambda t : (t[0], t[1][0]/t[1][1])).collect()
    curr_sse = sum(np.sum((cent[i] - p) ** 2) for (i, p) in result)
    for(i,p) in result:
        cent[i] = p
       
    iterl+=1

    
print("Centers:")
print(cent)

end = time.time()


# In[201]:


print("Total time : ", end="")
print(end-start)


# In[173]:



result_x = np.zeros(len(data_x.index))
j=0
for i in range(len(data_x.index)):
    result_x[i] = cal_distance(data_x.iloc[i].astype(float),cent)


# In[176]:


silhouette_avg = metrics.silhouette_score(data_x,result_x)
print(silhouette_avg)


# In[ ]:


spark.stop()

