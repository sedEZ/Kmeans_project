#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import random
import numpy as np
import pandas as pd
from sklearn import cluster, datasets
from sklearn.metrics import accuracy_score



# # Data Preprocessing

# In[ ]:


#data_x : features (Dataframe)
#data_y : labels   (np.array)
#label  : labels   (Dataframe)
iris = datasets.load_iris()
data_x = pd.DataFrame(iris.data)

data_y = iris.target

Simulate_time = 500


# In[ ]:


class our_k_means:
    def __init__(self,data_x,data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.label  = pd.DataFrame(data_y)
        self.acc    = 0
        
        #資料特性
        self.DCNT = len(self.data_x)               #資料個數
        self.DIM  = len(self.data_x.columns)       #資料維度
        self.K    = len(np.unique(self.data_y))    #叢聚個數
        self.MAX_ITER = 30                         #最大迭代
        self.MIN_PT = 0                            #最小變動點
        
        #k-means過程的參數
        self.data =[]
        self.cent =[]
        self.table=[]
        self.dis_k=[]
        self.cent_c=[]
        self.ch_pt = 0        
        self.iterl = 0 
        self.sse2 = 0
        
        #計算acc時的參數
        self.mass = 0
        self.origin_mass = 0
        self.closest = 0
        self.origin_label_table = np.arange(self.K)
        
        
    #run k-means    
    def run(self):
        self.data =[]
        self.cent =[]
        self.table=[]
        self.dis_k=[]
        self.cent_c=[]
        self.ch_pt = 0        
        self.iterl = 0 
        self.sse2 = 0
        
        #initialize tables
        self.data,self.cent = self.kmeans_init()   #初始化centroid
        for i in range(self.DCNT):
            self.table.append(0)
        for i in range(self.K):
            self.dis_k.append([])
            self.cent_c.append(0)
            for j in range(self.DIM):
                self.dis_k[i].append([])
        #first iteration        
        self.iterl = 0
        sse1 = -0.537
        self.sse2 = self.update_table()
        
        #update centroid & data clustering
        while self.iterl<self.MAX_ITER and sse1!=self.sse2 and self.ch_pt >self.MIN_PT  :
            sse1 = self.sse2
            self.iterl+=1
            self.update_cent()
            self.sse2 = self.update_table()
        
        self.table = pd.DataFrame(self.table)
        #self.print_result()
        
    #Calculate average accuracy    
    def calculate_acc(self,iterate_times):
        
        self.mass        = 0
        self.origin_mass = 0
        self.closest     = 0
        self.origin_label_table = np.arange(self.K)
        self.acc         = 0
        
        i = 0
        
        while( i < iterate_times):
            self.run()
            temp_acc = self.cal_1_acc()
            
            # Avoid the rare situations that some cluster are gone
            if len(np.unique(self.closest)) != self.K:
                continue
            
            i += 1
            self.acc += temp_acc
            
        self.acc /= i
        return self.acc

    
#---------------------------------------------------------------------------------
#----------------Subfunctions of calculate_acc(iterate_times)---------------------
#---------------------------------------------------------------------------------
    #all the clusters should be 1-D DataFrame which contains the same labels
    def find_mass(self,labels):
        mass      = np.zeros((self.K,self.DIM))
        num       = np.zeros(self.K)
        row_count = 0

        for i in labels.iloc[0:,0].values:
            for j in range(self.DIM):
                mass[i][j] += self.data_x.iloc[row_count][j]
            row_count += 1
            num[i] += 1

        for i in range(self.K):
            for j in range(self.DIM):
                mass[i][j] /= num[i]

        return mass   

    def calculate_closest(self):
        closest = np.zeros(self.K)
        for i in range(self.K):
            min_dist=float("inf")
            for j in range(self.K):
                dist = np.linalg.norm(self.mass[i]-self.origin_mass[j])
                if dist < min_dist:
                    min_dist = dist
                    closest[i]=j
        return closest 
    
    def relabel(self,origin_table,rename_table,target):
        target = target.replace(origin_table,rename_table)
        return target
    
    
    def cal_1_acc(self):    
        self.mass        = self.find_mass(self.table)
        self.origin_mass = self.find_mass(self.label)
        self.closest     = self.calculate_closest()
        self.table       = self.relabel(self.origin_label_table,self.closest,self.table)
                    
        return accuracy_score(self.label,self.table)
    
    
#---------------------------------------------------------------------------------
#------------------------------Subfunctions of run()------------------------------
#---------------------------------------------------------------------------------    
    def kmeans_init(self):
        data = self.data_x.values
        cent = []
        for i in range(self.DCNT):
            cent.append([])
            for j in range(self.DIM):
                cent[i].append([])
                
        pick = []
        counter = 0
        while(counter<self.K):
            rnd = random.randint(0,self.DCNT-1)
            if(rnd not in pick):
                pick.append(rnd)
                counter=counter+1
                
        for i in range(self.K):
            for j in range(self.DIM):
                cent[i][j] = data[pick[i]][j] 
        
        return data,cent
    
    def cal_distance(self,x,y):
        sum = 0
        for i in range(self.DIM):
            sum = sum + (self.data[x][i]-self.cent[y][i])*( self.data[x][i]-self.cent[y][i])
        return sum

            
    def update_table(self):
        t_sse = 0
        self.ch_pt = 0 
        for i in range(self.K):
            self.cent_c[i] = 0
            for j in range(self.DIM):
                self.dis_k[i][j] = 0
                
        for i in range(self.DCNT):
            min_dis = self.cal_distance(i,0)
            min_k=0
            for j in range(1,self.K):
                dis = self.cal_distance(i,j)
                if(dis<min_dis):
                    min_dis = dis
                    min_k = j
            self.ch_pt+=(self.table[i]!=min_k)
            self.table[i] = min_k
            self.cent_c[min_k] +=1
            t_sse+=min_dis
            for j in range(self.DIM):
                self.dis_k[min_k][j]+=self.data[i][j]
                
        return t_sse

    def update_cent(self):
        for i in range(self.K):
            for j in range(self.DIM):
                if self.cent_c[i] is not 0:
                    self.cent[i][j] = self.dis_k[i][j]/self.cent_c[i]
                else:
                    self.cent[i][j] = self.dis_k[i][j]

    def print_cent(self):
        for i in range(self.K):
            for j in range(self.DIM):
                print(self.cent[i][j],end=' ')
            print();

    def print_result(self):
        print("K means:")
        print(self.table)
        print("sse = ",end='')
        print(self.sse2)
        print("ch_pt = ",end='')
        print(self.ch_pt)
        print("iter = ",end='')
        print(self.iterl)     
    


# ## Here comes our k-means
# ### Let's run for 1 time and check the performance

# In[ ]:


result = our_k_means(data_x,data_y)
result.calculate_acc(1)


# ## Then , we run it for (Simulate_time) times

# In[ ]:


# Calculate
result.calculate_acc(Simulate_time)
result.acc


# # Let's run the k-means provides by sklearn
# ### -Then we can estimate how good we've done

# In[ ]:


def k_means_sklearn(datas,labels):
    # KMeans 演算法
    kmeans_fit = cluster.KMeans(n_clusters = 3).fit(datas)

    # 印出分群結果
    cluster_labels = kmeans_fit.labels_
    print("分群結果：")
    print(cluster_labels)
    print("---")
    
    # 印出品種看看
    print("True label：")
    print(labels)

    print("Number of samples: %d" % len(datas))
    
    return cluster_labels


# In[ ]:


#all the clusters should be 1-D DataFrame which contains the same labels
def find_mass(k,dim,table,data):
    mass = np.zeros((k,dim))
    num = np.zeros(k)
    row_count = 0

    for i in table.values:
        for j in range(dim):
            mass[i][j] += data.iloc[row_count][j]
        row_count += 1
        num[i] += 1
        
    for i in range(k):
        for j in range(dim):
            mass[i][j] /= num[i]
    
    return mass


# In[ ]:


def calculate_closest(k,origin,after_clustering):
    closest = np.zeros(k)
    for i in range(k):
        min_dist=float("inf")
        for j in range(k):
            dist = np.linalg.norm(after_clustering[i]-origin[j])
            if dist < min_dist:
                min_dist = dist
                closest[i]=j
    return closest 


# In[ ]:


def relabel(origin_table,rename_table,target):
    target = target.replace(origin_table,rename_table)
    return target


# In[ ]:


DIM  = len(data_x.columns)       #資料維度
K    = len(np.unique(data_y))    #叢聚個數
cluster_labels = k_means_sklearn(data_x,data_y) # sklearn.cluster.k_means_sklearn


# In[ ]:


#A list that can be used to compared the order of label
temp = np.arange(K)
#Turn the labels trained by skilearn into DataFrame format
cluster_labels = pd.DataFrame(cluster_labels)

#Find the mass of data with trained labels
mass_sklean_kmeans = find_mass(K,DIM,cluster_labels.iloc[0:,0],data_x)
#Find tha mass of data with original labels
mass_origin        = find_mass(K,DIM,label.iloc[0:,0],data_x)
#Fine the correct cluster names & Relabel
closest_sklearn    = calculate_closest(K,mass_origin,mass_sklean_kmeans).astype(int)
cluster_labels     = relabel(temp,closest_sklearn ,cluster_labels)


# In[ ]:


#Valid accuracy
sklearn_acc = accuracy_score(label,cluster_labels)
sklearn_acc


# In[ ]:




