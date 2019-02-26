'''
-see each genre's movies  done
-random sampling 10 movies in the three genre done
-plot 
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

movie = np.loadtxt('data/movies.txt').astype('string')

with open("./data/movies.txt", encoding="mac_roman") as f:
    metadata = []
    while True:
        movie = f.readline()
        if movie == "": break

        movie = [eval(v) if i != 1 else v for i, v in enumerate(movie.strip(" ").split("\t"))]
        metadata.append(movie)
        
        
# divide it as ndarray of feature and a vector of name       
feat_mat = []
for entry in metadata:
    feat_mat.append(entry[2:])
feat_mat = np.asarray(feat_mat) 

mv_name = []
for entry in metadata:
    mv_name.append(entry[1])
    
    
# choose genre Film-noir children musical   
# column index [10, 4, 12]
# find index array
    
ind1 = np.where(feat_mat[:,10]==1)[0]  # film noir
ind2 = np.where(feat_mat[:,4]==1)[0]   # children
ind3 = np.where(feat_mat[:,12]==1)[0]  # musical
# random select 10 films from each genre
sel_ind1 = ind1[np.random.permutation(len(ind1))[:10]]
sel_ind2 = ind2[np.random.permutation(len(ind2))[:10]]
sel_ind3 = ind3[np.random.permutation(len(ind3))[:10]]
for i in sel_ind1:
    print(mv_name[i])
for i in sel_ind2:
    print(mv_name[i])
for i in sel_ind3:
    print(mv_name[i])


