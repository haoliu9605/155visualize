"""
Created on Mon Feb 25 21:39:59 2019

@author: liuhao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import collections

sns.set(style="darkgrid", font_scale=0.9)

def projection(U, V):
    """
    U: N x K numpy array
    V: M x K numpy array
    """
    # Normalization in K-D
    if U is not None:
        U = U.transpose()
    normV = V - np.mean(V, axis=0)
    normV = normV.transpose()

    # Projection to 2D
    u, s, vh = np.linalg.svd(normV)
    proj = u[:, :2]
    if U is not None:
        lowdimU = np.matmul(proj.transpose(), U).transpose()
    lowdimV = np.matmul(proj.transpose(), normV).transpose()

    # Final normalization in 2D
    if U is not None:
        repreU = (lowdimU - np.mean(lowdimU, axis=0)) / np.std(lowdimU, axis=0)
    else:
        repreU = None
    repreV = (lowdimV - np.mean(lowdimV, axis=0)) / np.std(lowdimV, axis=0)
    return repreU, repreV

# Read data
with open("./data/data.txt") as f:
    Y = []
    while True:
        rating = f.readline()
        if rating == "": break

        Yij = [eval(v) for v in rating.strip(" ").split("\t")]
        Y.append(Yij)

# Calculate statistics
user_count = np.zeros(943)
movie_count = np.zeros(1682)
movie_rate = np.zeros(1682)
for Yij in Y:
    user_count[Yij[0] - 1] += 1
    movie_count[Yij[1] - 1] += 1
    movie_rate[Yij[1] - 1] += Yij[2]
movie_rate = movie_rate / movie_count

# Read meta-data
with open("./data/movies.txt", encoding="mac_roman") as f:
    metadata = []
    while True:
        movie = f.readline()
        if movie == "": break

        movie = [eval(v) if i != 1 else v for i, v in enumerate(movie.strip(" ").split("\t"))]
        metadata.append(movie)

# Create Pandas Dataframe
metadf = pd.DataFrame(data=metadata, columns=["movie ID", "movie title", "Unknown", "Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
"Sci-Fi", "Thriller", "War", "Western"])
metadf['lat_vec'] = pd.Series(list(V), index=metadf.index)
repreU, repreV = projection(U, V)
metadf['x'] = pd.Series(repreV[:, 0], index=metadf.index)
metadf['y'] = pd.Series(repreV[:, 1], index=metadf.index)
   # Visualize selected movie
some_movie = [222, 228, 59, 60, 61, 185, 127, 616, 542, 553]
metadf['selected'] = pd.Series([id in some_movie for id in metadf['movie ID']], index=metadf.index)
some_movie = metadf[metadf['selected']==True]
print(some_movie)

plt.figure()
sns_plot = sns.scatterplot(x='x', y='y', data=some_movie, s=40)
for i, txt in enumerate(some_movie['movie title']):
    print(i)
    name = txt.strip("\"")[:-7].strip(" ")
    print(name, some_movie['x'].iloc[i], some_movie['y'].iloc[i])
    if i == 4:
        sns_plot.annotate(name, (some_movie['x'].iloc[i] - len(name) * 0.028, 
                             some_movie['y'].iloc[i]+0.08))
    elif i == 6:
        sns_plot.annotate(name, (some_movie['x'].iloc[i] - len(name) * 0.028+0.9, 
                             some_movie['y'].iloc[i]+0.02))        
    elif i == 7:
        sns_plot.annotate(name, (some_movie['x'].iloc[i] - len(name) * 0.028+0.2, 
                             some_movie['y'].iloc[i]+0.08))        
    elif i == 8:
        sns_plot.annotate(name, (some_movie['x'].iloc[i] - len(name) * 0.028+0.8, 
                             some_movie['y'].iloc[i]+0.08))        
    else:
        sns_plot.annotate(name, (some_movie['x'].iloc[i] - len(name) * 0.028, 
                             some_movie['y'].iloc[i]+0.04))
fig = sns_plot.get_figure()
fig.savefig("all_scatter_selected.png")
plt.close()