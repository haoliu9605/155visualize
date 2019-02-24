import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils_5B_adv import train_model, get_err

best_reg = 0.1
best_eta = 0.015

Y_train = np.loadtxt('data/train.txt').astype(int)
Y_test = np.loadtxt('data/test.txt').astype(int)

M = max(max(Y_train[:,0]), max(Y_test[:,0])).astype(int) # users
N = max(max(Y_train[:,1]), max(Y_test[:,1])).astype(int) # movies
print("Factorizing with ", M, " users, ", N, " movies.")
print("Factorizing with ", M, " users, ", N, " movies.")
K = 20

#normalize / -mean
Ylen = Y_train.shape[0]
ind = list(range(Ylen))
Ymean = np.mean(Y_train[:, 2])
for t in ind:
    Y_train[t][2] = Y_train[t][2]-Ymean
for t in list(range(Y_test.shape[0])):
    Y_test[t][2] = Y_test[t][2]-Ymean
print(Ymean)

# Perform CV when best param is not known
if best_reg == -1 and best_eta == -1:
    reg = [0.05, 0.1, 0.2, 0.4, 1]
    eta = [0.015, 0.03, 0.06] # learning rate
    E_out = np.zeros((len(reg),len(eta)))

    # Use to compute Ein and Eout
    for r in range(len(reg)):
        for e in range(len(eta)):
            U, V, A, B, err = train_model(M, N, K, eta[e], reg[r], Y_train)
            err_out = get_err(U, V, Y_test, A, B)
            E_out[r, e] = err_out
            print(r, e, err_out)

            if (best_reg == -1 and best_eta == -1) or E_out[best_reg, best_eta] > err_out:
                best_reg, best_eta = r, e

    data = pd.DataFrame(E_out, index = np.array(reg), columns = np.array(eta))
    sns.set()
    ax = sns.heatmap(data,cmap="YlGnBu",annot=True,fmt='.3g')
    fig = ax.get_figure()
    fig.savefig('heatmap_5B_adv.png')

U, V, A, B, err = train_model(M, N, K, best_eta, best_reg, Y_train)
normV = V - np.mean(V, axis=0)

U = U.transpose()
normV = normV.transpose()

u, s, vh = np.linalg.svd(normV)
proj = u[:, :2]
lowdimU = np.matmul(proj.transpose(), U).transpose()
lowdimV = np.matmul(proj.transpose(), normV).transpose()

repreU = (lowdimU - np.mean(lowdimU, axis=0)) / np.std(lowdimU, axis=0)
repreV = (lowdimV - np.mean(lowdimV, axis=0)) / np.std(lowdimV, axis=0)

with open("./data/data.txt") as f:
    Y = []
    while True:
        rating = f.readline()
        if rating == "": break

        Yij = [eval(v) for v in rating.strip(" ").split("\t")]
        Y.append(Yij)

with open("./data/movies.txt", encoding="mac_roman") as f:
    metadata = []
    while True:
        movie = f.readline()
        if movie == "": break

        movie = [eval(v) if i != 1 else v for i, v in enumerate(movie.strip(" ").split("\t"))]
        metadata.append(movie)

Y = np.array(Y)
pdY = pd.DataFrame(data=Y, columns=["user ID", "movie ID", "rating"])

user_count = np.zeros(943)
movie_count = np.zeros(1682)
movie_rate = np.zeros(1682)
for Yij in Y:
    user_count[Yij[0] - 1] += 1
    movie_count[Yij[1] - 1] += 1
    movie_rate[Yij[1] - 1] += Yij[2]
movie_rate = movie_rate / movie_count

metadf = pd.DataFrame(data=metadata, columns=["movie ID", "movie title", "Unknown", "Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
"Sci-Fi", "Thriller", "War", "Western"])

metadf['x'] = pd.Series(repreV[:, 0], index=metadf.index)
metadf['y'] = pd.Series(repreV[:, 1], index=metadf.index)

popular_movie = np.argsort(movie_count)[::-1]
popular_movie = popular_movie[:10]
metadf['popular'] = pd.Series([id in popular_movie for id in metadf['movie ID']-1], index=metadf.index)
popular_movie = metadf[metadf['popular']==True]

plt.figure()
sns_plot = sns.scatterplot(x='x', y='y', data=popular_movie, s=30)
for i, txt in enumerate(popular_movie['movie title']):
    sns_plot.annotate(txt, (popular_movie['x'].iloc[i], popular_movie['y'].iloc[i]))
fig = sns_plot.get_figure()
fig.savefig("all_scatter.png")
plt.close()
