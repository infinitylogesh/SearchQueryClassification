import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve
import numpy as np

# Plots a 3d scatter plot by doing PCA on the matrix to 3D matrix
# TODO:Generalize the terms
def plot3D(texts,queries_matrix,Y):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(queries_matrix)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],c=Y,cmap=plt.cm.Paired)
    # for i in xrange(0,Y.size):
    #     ax.scatter(X_reduced[:, 0][i], X_reduced[:, 1][i], X_reduced[:, 2][i],
    #
    #     ax.text(X_reduced[:, 0][i], X_reduced[:, 1][i], X_reduced[:, 2][i] , '%s' % (texts[i]), size=5, zorder=1)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels(())
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels(())
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels(())
    plt.show()

# Plots learning curve of an estimator
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def word_score_plot(wordScoreTupleArray,topLimit):
    ax = plt.subplot(111)
    width=0.3
    scores = zip(*wordScoreTupleArray[:topLimit])[0]
    bins = map(lambda x: x-width/2,range(1,len(scores)+1))
    ax.bar(bins,scores,width=width)
    ax.set_xticks(map(lambda x: x, range(1,len(scores)+1)))
    ax.set_xticklabels(zip(*wordScoreTupleArray[:topLimit])[1],rotation=45)
    plt.ylim(0,round(max(scores)))
    plt.show()


