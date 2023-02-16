import sklearn
from sklearn.cluster import KMeans
import numpy as np
from multiprocessing import Pool
import os 
import torch

import faiss

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X):
        #print(X.shape)
        #X = torch.Tensor(X)
        #print(type(X))
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init) # 
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


def run(ncl,max_iter,sampl,clusteringID): 
    C = KMeans(n_clusters=ncl,max_iter=max_iter,n_init=1,verbose=False).fit(sampl)
    print('Clustering ID = ',clusteringID,'n_iter = ',C.n_iter_,'Inertia = ',C.inertia_)
    return C 


def kmeans_with_multiple_runs(ncl,max_iter,nclustering,sampl):
    
    num_processors = os.cpu_count() -1
    p=Pool(processes = num_processors)
          
    args = [] 
        
    for i in range(nclustering): 
        args.append([ncl,max_iter,sampl,i])
    clusters = p.starmap(run,args)

    inertias = []
    for i in range(len(clusters)):
        inertias.append(clusters[i].inertia_)
    index = inertias.index(min(inertias))

    print('The best inertia = ', min(inertias))

    p.close()
    p.join()

    return clusters[index], min(inertias)
    
def kmeans_with_multiple_runs_basic(ncl,max_iter,nclustering,sampl):
    
    #num_processors = os.cpu_count() -1
    #p=Pool(processes = num_processors)

    inertias = []
    cluster_models = []
    
    for i in range(nclustering): 
        cluster_model = run(ncl=ncl,max_iter=max_iter,sampl=sampl, clusteringID=i)
        inertias.append(cluster_model.inertia_)
        cluster_models.append(cluster_model)

    #args = []
    #clusters = p.starmap(run,args)
    #for i in range(nclustering): 
        #args.append([ncl,max_iter,sampl,i])
    #clusters = p.starmap(run,args)
    #inertias = []
    #for i in range(len(clusters)):
        #inertias.append(clusters[i].inertia_)
    
    index = inertias.index(min(inertias))
    print('The best inertia = ', min(inertias))

    #p.close()
    #p.join()

    return cluster_models[index], min(inertias)

def kmeans_with_multiple_runs_faiss(ncl,max_iter,nclustering,sampl):

    inertias = []
    cluster_models = []
    KMmodel = FaissKMeans(n_clusters=ncl, n_init=nclustering, max_iter=max_iter)
    KMmodel.fit(sampl)
    print(f"Run: {i} Inertia: {KMmodel.inertia_}")
    inertia = KMmodel.inertia_
    cluster_model = KMmodel

    #for i in range(nclustering): 
        #KMmodel = FaissKMeans(n_clusters=ncl, n_init=i, max_iter=max_iter)
        #KMmodel.fit(sampl)
        #print(f"Run: {i} Inertia: {KMmodel.inertia_}")
        #inertias.append(KMmodel.inertia_)
        #cluster_models.append(KMmodel)
    
    #index = inertias.index(min(inertias))
    print('The best inertia = ', inertia)

    # cluster_models[index], min(inertias)

    return KMmodel, inertia
