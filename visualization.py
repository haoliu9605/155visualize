import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections

sns.set(style="darkgrid", font_scale=1.4)

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

def kde_visualize(repreU, repreV, Y, movie_ID):
    movie_ID = set(movie_ID)
    data = collections.defualtdict(list)
    for item in Y:
        if item[1] in movie_ID:
            data[item[1]].append(item[0])
    plt.figure()
    for movie,users in data.items():
        y,x = [],[]
        for u in users:
            y.append(repreU[u][0])
            x.append(repreU[u][1])
        sns_plot = sns.kdeplot(x, y)#cmap="Reds", shade=True, shade_lowest=False)
    fig = sns_plot.get_figure()
    fig.savefig("kde.png")
    plt.close()
    return

def visualize(U, V):
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

    # Visualize audience for movies
    kde_visualize(repreU, repreV, Y, [222, 228, 59, 60, 61, 185, 127, 616, 542, 553])

    # Visualize different genre
    genre_vec = []
    all_genre = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    for genre in all_genre:
        all_movie_in_genre = metadf[metadf[genre] == True]
        mean_v = np.mean(all_movie_in_genre['lat_vec'].values)
        genre_vec.append(mean_v)
    _, genre_vec = projection(None, np.array(genre_vec))
    genre_df = pd.DataFrame(data=genre_vec, columns=["x", "y"])

    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=genre_df, s=30)
    for i, txt in enumerate(all_genre):
        sns_plot.annotate(txt, (genre_df['x'].iloc[i], genre_df['y'].iloc[i]))
    fig = sns_plot.get_figure()
    fig.savefig("all_scatter_genre.png")
    plt.close()

    # Visualize selected movie
    some_movie = [222, 228, 59, 60, 61, 185, 127, 616, 542, 553]
    metadf['selected'] = pd.Series([id in some_movie for id in metadf['movie ID']], index=metadf.index)
    some_movie = metadf[metadf['selected']==True]
    print(some_movie)

    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=some_movie, s=40)
    for i, txt in enumerate(some_movie['movie title']):
        name = txt.strip("\"")[:-7].strip(" ")
        print(name, some_movie['x'].iloc[i], some_movie['y'].iloc[i])
        sns_plot.annotate(name, (max(some_movie['x'].iloc[i] - len(name) * 0.028, -2.35), some_movie['y'].iloc[i]+0.04))
    fig = sns_plot.get_figure()
    fig.savefig("all_scatter_selected.png")
    plt.close()

    # Visualize popular movie
    popular_movie = np.argsort(movie_count)[::-1]
    popular_movie = popular_movie[:10]
    metadf['popular'] = pd.Series([id in popular_movie for id in metadf['movie ID']-1], index=metadf.index)
    popular_movie = metadf[metadf['popular']==True]
    print(popular_movie)

    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=popular_movie, s=40)
    for i, txt in enumerate(popular_movie['movie title']):
        sns_plot.annotate(txt.strip("\"")[:-7], (popular_movie['x'].iloc[i], popular_movie['y'].iloc[i]+0.03))
    fig = sns_plot.get_figure()
    fig.savefig("all_scatter_popular.png")
    plt.close()

    # Visualize best movie
    trimmed_movie_rate = [x if movie_count[i] > 50 else 0 for i, x in enumerate(movie_rate)]
    best_movie = np.argsort(trimmed_movie_rate)[::-1]
    best_movie = best_movie[:10]
    metadf['best'] = pd.Series([id in best_movie for id in metadf['movie ID']-1], index=metadf.index)
    best_movie = metadf[metadf['best']==True]
    print(best_movie)

    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=best_movie, s=40)
    for i, txt in enumerate(best_movie['movie title']):
        sns_plot.annotate(txt.strip("\"")[:-7], (best_movie['x'].iloc[i], best_movie['y'].iloc[i]+0.03))
    fig = sns_plot.get_figure()
    fig.savefig("all_scatter_best.png")
    plt.close()
