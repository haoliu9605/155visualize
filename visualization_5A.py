import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import collections

sns.set(style="darkgrid", font_scale=1.0)

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

def kde_visualize(U, V, Y, movie_ID, metadf, name):
    data = collections.defaultdict(list)
    for item in Y:
        if item[1] in movie_ID and item[2] >= 4:
            data[item[1]].append(item[0])
    f = plt.figure(figsize=(15, 15))
    #plt.axis('off')
    i = 1
    for movie in movie_ID:
        users = data[movie]
        y,x = [],[]
        for u in users:
            x.append(U[u-1][0])
            y.append(U[u-1][1])
        f.add_subplot(3,3,i,xlim = (-3,3), ylim = (-3,3))
        sns.kdeplot(x, y, cmap="Blues", shade=True, shade_lowest=False)
        plt.scatter(V[movie-1][0], V[movie-1][1], marker='o', s=100,color='orange')
        plt.tick_params(labelsize=15)
        plt.xlabel('x1',size = 20)
        plt.ylabel('x2',size = 20)
        #plt.ylim((-3, 3))
        #plt.xlim((-3, 3))
        plt.title(str(metadf["movie title"].iloc[movie-1]),fontsize=20)

        i += 1
    plt.tight_layout()
    plt.savefig(name + '/kde.png')
    plt.close()
    return


def rend_sam_movie_ingenre():
    #movie = np.loadtxt('data/movies.txt').astype('string')

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
    np.random.seed(170)
    sel_ind1 = ind1[np.random.permutation(len(ind1))[:10]]
    sel_ind2 = ind2[np.random.permutation(len(ind2))[:10]]
    sel_ind3 = ind3[np.random.permutation(len(ind3))[:10]]
    for i in range(len(sel_ind1)):
        sel_ind1[i] += 1
    for i in range(len(sel_ind2)):
        sel_ind2[i] += 1
    for i in range(len(sel_ind3)):
        sel_ind3[i] += 1
    print("selected film noir ")
    for i in sel_ind1:
        print(mv_name[i-1])
    print("selected children ")
    for i in sel_ind2:
        print(mv_name[i-1])
    print("selected musical ")
    for i in sel_ind3:
        print(mv_name[i-1])
    return sel_ind1, sel_ind2, sel_ind3

def visualize(U, V, dirname):
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
    kde_visualize(repreU, repreV, Y, [222, 228, 542, 59, 60, 61, 185, 616, 127], metadf, dirname)

    # Visualize different genre
    genre_vec = []
    all_genre = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    for genre in all_genre:
        all_movie_in_genre = metadf[metadf[genre] == True]
        mean_v = np.mean(all_movie_in_genre['lat_vec'].values)
        print(genre, np.mean(all_movie_in_genre['x'].values), np.mean(all_movie_in_genre['y'].values))
        genre_vec.append(mean_v)
    _, genre_vec = projection(None, np.array(genre_vec))
    genre_df = pd.DataFrame(data=genre_vec, columns=["x", "y"])

    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=genre_df, s=30)
    for i, txt in enumerate(all_genre):
        #print(txt," i ",i)
        if i == 6:
            sns_plot.annotate(txt, (genre_df['x'].iloc[i]-1.0, genre_df['y'].iloc[i]))
        elif i == 12:
            sns_plot.annotate(txt, (genre_df['x'].iloc[i]-0.5, genre_df['y'].iloc[i]))
        elif i == 7:
            sns_plot.annotate(txt, (genre_df['x'].iloc[i]-0.5, genre_df['y'].iloc[i]))
        else:
            sns_plot.annotate(txt, (genre_df['x'].iloc[i], genre_df['y'].iloc[i]))
    fig = sns_plot.get_figure()
    fig.savefig(dirname + "/all_scatter_genre.png")
    plt.close()

    # Visualize genre movies
    sns.set(style="ticks", color_codes=True)
    plt.figure()
    selected_genre = ["Film-Noir", "Children\'s", "Musical"]
    colors = ["Blues", "Reds", "Greens"]
    movie_in_genre = metadf[(metadf["Musical"] == True) | (metadf["Film-Noir"] == True) | (metadf["Children\'s"] == True)]
    g = sns.JointGrid(x="x", y="y", data=movie_in_genre)
    for genre, color in zip(selected_genre, colors):
        all_movie_in_genre = metadf[metadf[genre] == True]
        #sns_plot = sns.scatterplot(x='x', y='y', data=all_movie_in_genre, label=genre)
        x_list, y_list = [], []
        for i in range(len(all_movie_in_genre)):
            x_list.append(all_movie_in_genre['x'].iloc[i])
            y_list.append(all_movie_in_genre['y'].iloc[i])
        custom_color_map = LinearSegmentedColormap.from_list(
            name='custom',
            colors=[color[0]]*10,
        )
        sns_plot = sns.kdeplot(x_list, y_list, label=genre, n_levels=7, shade=False, shade_lowest=False, cut=2, ax=g.ax_joint, cmap=custom_color_map, alpha=0.7)
        sns.distplot(x_list, kde=True, hist=False, color=color[0], ax=g.ax_marg_x, kde_kws={"shade":True})
        sns.distplot(y_list, kde=True, hist=False, color=color[0], ax=g.ax_marg_y, vertical=True, kde_kws={"shade":True})
    fig = sns_plot.get_figure()
    sns_plot.legend()
    fig.savefig(dirname + "/all_scatter_selected_genre.png")
    plt.close()
    sns.set(style="darkgrid", font_scale=1.0)

    #  Visualize random sampling movie in three genres
    sel_ind1, sel_ind2, sel_ind3 = rend_sam_movie_ingenre()
    # genre1
    some_movie = sel_ind1
    metadf['selected'] = pd.Series([id in some_movie for id in metadf['movie ID']], index=metadf.index)
    some_movie = metadf[metadf['selected']==True]
    print(some_movie)
    #genre 1
    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=some_movie, s=40)
    for i, txt in enumerate(some_movie['movie title']):
        name = txt.strip("\"")[:-7].strip(" ")
        #print(name, i, some_movie['x'].iloc[i], some_movie['y'].iloc[i])
        if i == 0:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i]-0.5, some_movie['y'].iloc[i]))
        elif i == 2:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i]-0.55, some_movie['y'].iloc[i]-0.1))
        elif i == 3:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i]-0.55, some_movie['y'].iloc[i]-0.05))
        else:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i], some_movie['y'].iloc[i]))
    fig = sns_plot.get_figure()
    fig.savefig(dirname + "/rand_selected_genre1.png")
    plt.close()
    # genre2
    some_movie = sel_ind2
    metadf['selected'] = pd.Series([id in some_movie for id in metadf['movie ID']], index=metadf.index)
    some_movie = metadf[metadf['selected']==True]
    print(some_movie)
    #genre 2
    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=some_movie, s=40)
    for i, txt in enumerate(some_movie['movie title']):
        name = txt.strip("\"")[:-7].strip(" ")
        #print(name, some_movie['x'].iloc[i], some_movie['y'].iloc[i])
        sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i], some_movie['y'].iloc[i]))
    fig = sns_plot.get_figure()
    fig.savefig(dirname + "/rand_selected_genre2.png")
    plt.close()
    # genre3
    some_movie = sel_ind3
    metadf['selected'] = pd.Series([id in some_movie for id in metadf['movie ID']], index=metadf.index)
    some_movie = metadf[metadf['selected']==True]
    print(some_movie)
    #genre 3
    plt.figure()
    sns_plot = sns.scatterplot(x='x', y='y', data=some_movie, s=40)
    for i, txt in enumerate(some_movie['movie title']):
        name = txt.strip("\"")[:-7].strip(" ")
        #print(name, i, some_movie['x'].iloc[i], some_movie['y'].iloc[i])
        if i == 7:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i]-0.75, some_movie['y'].iloc[i]))
        elif i == 8:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i]-0.65, some_movie['y'].iloc[i]))
        elif i == 2:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i]-1.2, some_movie['y'].iloc[i]))
        elif i == 9:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i]-2.2, some_movie['y'].iloc[i]+0.05))
        elif i == 1:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i]-0.8, some_movie['y'].iloc[i]))
        else:
            sns_plot.annotate(txt.strip("\"")[:-7], (some_movie['x'].iloc[i], some_movie['y'].iloc[i]))
        
    fig = sns_plot.get_figure()
    fig.savefig(dirname + "/rand_selected_genre3.png")
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
        print(i, ' present i ',name, some_movie['x'].iloc[i], some_movie['y'].iloc[i])
        if i == 0:
            sns_plot.annotate(name, (some_movie['x'].iloc[i]-0.7, some_movie['y'].iloc[i]))
        elif i == 1:
            sns_plot.annotate(name, (some_movie['x'].iloc[i]-0.7, some_movie['y'].iloc[i]))
        elif i == 2:
            sns_plot.annotate(name, (some_movie['x'].iloc[i]-0.75, some_movie['y'].iloc[i]))
        elif i == 3:
            sns_plot.annotate(name, (some_movie['x'].iloc[i]-0.6, some_movie['y'].iloc[i]))
        elif i == 4:
            sns_plot.annotate(name, (some_movie['x'].iloc[i]-0.3, some_movie['y'].iloc[i]))
        elif i == 6:
            sns_plot.annotate(name, (some_movie['x'].iloc[i]-0.3, some_movie['y'].iloc[i]+0.1))
        elif i == 9:
            sns_plot.annotate(name, (some_movie['x'].iloc[i]-0.7, some_movie['y'].iloc[i]+0.1))
        else:
            sns_plot.annotate(name, (some_movie['x'].iloc[i], some_movie['y'].iloc[i]))
    fig = sns_plot.get_figure()
    fig.savefig(dirname + "/all_scatter_selected.png")
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
    fig.savefig(dirname + "/all_scatter_popular.png")
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
        #print(txt, ' present i ', i)
        if i == 4:
            sns_plot.annotate(txt.strip("\"")[:-7], (best_movie['x'].iloc[i]-0.15, best_movie['y'].iloc[i]+0.05))
        elif i == 9:
            sns_plot.annotate(txt.strip("\"")[:-7], (best_movie['x'].iloc[i]-0.1, best_movie['y'].iloc[i]))
        elif i == 3:
            sns_plot.annotate(txt.strip("\"")[:-7], (best_movie['x'].iloc[i]-0.42, best_movie['y'].iloc[i]-0.03))
        elif i == 8:
            sns_plot.annotate(txt.strip("\"")[:-7], (best_movie['x'].iloc[i]-0.02, best_movie['y'].iloc[i]-0.05))
        elif i == 7:
            sns_plot.annotate(txt.strip("\"")[:-7], (best_movie['x'].iloc[i]-0.03, best_movie['y'].iloc[i]+0.05))
        elif i == 5:
            sns_plot.annotate(txt.strip("\"")[:-7], (best_movie['x'].iloc[i]-0.03, best_movie['y'].iloc[i]))
        else:
            sns_plot.annotate(txt.strip("\"")[:-7], (best_movie['x'].iloc[i], best_movie['y'].iloc[i]+0.04))
    fig = sns_plot.get_figure()
    fig.savefig(dirname + "/all_scatter_best.png")
    plt.close()
