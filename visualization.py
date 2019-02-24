import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize(U, V):
    # Normalization in K-D
    U = U.transpose()
    normV = V - np.mean(V, axis=0)
    normV = normV.transpose()

    # Projection to 2D
    u, s, vh = np.linalg.svd(normV)
    proj = u[:, :2]
    lowdimU = np.matmul(proj.transpose(), U).transpose()
    lowdimV = np.matmul(proj.transpose(), normV).transpose()

    # Final normalization in 2D
    repreU = (lowdimU - np.mean(lowdimU, axis=0)) / np.std(lowdimU, axis=0)
    repreV = (lowdimV - np.mean(lowdimV, axis=0)) / np.std(lowdimV, axis=0)

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
    pdY = pd.DataFrame(data=np.array(Y), columns=["user ID", "movie ID", "rating"])
    metadf = pd.DataFrame(data=metadata, columns=["movie ID", "movie title", "Unknown", "Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"])

    metadf['x'] = pd.Series(repreV[:, 0], index=metadf.index)
    metadf['y'] = pd.Series(repreV[:, 1], index=metadf.index)

    popular_movie = np.argsort(movie_count)[::-1]
    popular_movie = popular_movie[:20]
    metadf['popular'] = pd.Series([id in popular_movie for id in metadf['movie ID']-1], index=metadf.index)
    popular_movie = metadf[metadf['popular']==True]

    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=popular_movie, s=30)
    for i, txt in enumerate(popular_movie['movie title']):
        sns_plot.annotate(txt, (popular_movie['x'].iloc[i], popular_movie['y'].iloc[i]))
    fig = sns_plot.get_figure()
    fig.savefig("all_scatter.png")
    plt.close()
