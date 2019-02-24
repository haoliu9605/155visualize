import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(style="dark", font_scale=1.4)

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

plt.figure()
sns_plot1 = sns.countplot(Y[:, 2], palette="coolwarm")
fig = sns_plot1.get_figure()
fig.savefig("all_rating.png")

plt.figure()
popular_movie = np.argsort(movie_count)[::-1]
popular_movie = popular_movie[:10]
popular_Y = pdY[[(x-1 in popular_movie) for x in pdY["movie ID"]]]
sns_plot2 = sns.countplot(x="rating", hue="movie ID", data=popular_Y, palette="coolwarm")
plt.legend(loc="upper left", title="movie ID")
fig = sns_plot2.get_figure()
fig.savefig("pop_movie_rating_detail1.png")

plt.figure()
sns_plot3 = sns.countplot(x="movie ID", hue="rating", data=popular_Y, palette="coolwarm")
plt.legend(loc="upper right", title="rating")
fig = sns_plot3.get_figure()
fig.savefig("pop_movie_rating_detail2.png")

plt.figure()
sns_plot = sns.countplot(popular_Y["rating"], palette="coolwarm")
fig = sns_plot.get_figure()
fig.savefig("pop_movie_rating.png")

plt.figure()
best_movie = np.argsort(movie_rate)[::-1]
best_movie = best_movie[:10]
best_Y = pdY[[(x-1 in best_movie) for x in pdY["movie ID"]]]
sns_plot4 = sns.countplot(x="rating", hue="movie ID", data=best_Y, palette="coolwarm", order=[1, 2, 3, 4, 5])
plt.legend(loc="upper left", title="movie ID")
fig = sns_plot4.get_figure()
fig.savefig("best_movie_rating_detail.png")

plt.figure()
sns_plot = sns.countplot(best_Y["rating"], palette="coolwarm", order=[1, 2, 3, 4, 5])
fig = sns_plot.get_figure()
fig.savefig("best_movie_rating.png")

plt.figure()
movie_rate = [x if movie_count[i] > 20 else 0 for i, x in enumerate(movie_rate)]
best_movie = np.argsort(movie_rate)[::-1]
best_movie = best_movie[:10]
best_Y = pdY[[(x-1 in best_movie) for x in pdY["movie ID"]]]
sns_plot5 = sns.countplot(x="rating", hue="movie ID", data=best_Y, palette="coolwarm")
plt.legend(loc="upper left", title="movie ID")
fig = sns_plot5.get_figure()
fig.savefig("best_movie_rating_thresh.png")

plt.figure()
horror_movie = set([id for id in range(1682) if metadata[id][13] == 1])
horror_Y = pdY[[(x-1 in horror_movie) for x in pdY["movie ID"]]]
sns_plot = sns.countplot(horror_Y["rating"], palette="coolwarm", order=[1, 2, 3, 4, 5])
fig = sns_plot.get_figure()
fig.savefig("horror_movie_rating.png")

plt.figure()
animation_movie = set([id for id in range(1682) if metadata[id][5] == 1])
animation_Y = pdY[[(x-1 in animation_movie) for x in pdY["movie ID"]]]
sns_plot = sns.countplot(animation_Y["rating"], palette="coolwarm", order=[1, 2, 3, 4, 5])
fig = sns_plot.get_figure()
fig.savefig("animation_movie_rating.png")

plt.figure()
scifi_movie = set([id for id in range(1682) if metadata[id][17] == 1])
scifi_Y = pdY[[(x-1 in scifi_movie) for x in pdY["movie ID"]]]
sns_plot = sns.countplot(scifi_Y["rating"], palette="coolwarm", order=[1, 2, 3, 4, 5])
fig = sns_plot.get_figure()
fig.savefig("scifi_movie_rating.png")

plt.figure()
war_movie = set([id for id in range(1682) if metadata[id][-2] == 1])
war_Y = pdY[[(x-1 in war_movie) for x in pdY["movie ID"]]]
sns_plot = sns.countplot(war_Y["rating"], palette="coolwarm", order=[1, 2, 3, 4, 5])
fig = sns_plot.get_figure()
fig.savefig("war_movie_rating.png")

plt.figure()
musical_movie = set([id for id in range(1682) if metadata[id][14] == 1])
musical_Y = pdY[[(x-1 in musical_movie) for x in pdY["movie ID"]]]
sns_plot = sns.countplot(musical_Y["rating"], palette="coolwarm", order=[1, 2, 3, 4, 5])
fig = sns_plot.get_figure()
fig.savefig("musical_movie_rating.png")
