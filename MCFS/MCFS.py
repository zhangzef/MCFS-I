from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding, Isomap
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def mcfs(X, n_selected_features, i, n_emb, n_neighbors):
    """
    This function implements unsupervised feature selection for multi-cluster data.

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_selected_features: {int}
        number of features to select
    i: {int: 0, 1}
        0: use MCFS
        1: use MCFS-I
    n_emb: {int}
        The dimension of the projected subspace.
    n_neighbors : int,
        0: default setting

    Output
    ------
    W: {numpy array}, shape(n_features, n_clusters)
        feature weight matrix
    """

    n_sample, n_feature = X.shape

    if i == 0:
        if n_neighbors==0:
            spe = SpectralEmbedding(n_components=n_emb)
        else:
            spe = SpectralEmbedding(n_components=n_emb, n_neighbors=n_neighbors)
        Y = spe.fit_transform(X)
    elif i == 1:
        if n_neighbors==0:
            iso = Isomap(n_components=n_emb)
        else:
            iso = Isomap(n_components=n_emb, n_neighbors=n_neighbors)
        Y = iso.fit_transform(X)

    # solve K L1-regularized regression problem using LARs algorithm with cardinality constraint being d
    W = np.zeros((n_feature, n_emb))
    for i in range(n_emb):
        clf = linear_model.Lars(n_nonzero_coefs=n_selected_features)
        clf.fit(X, Y[:, i])
        W[:, i] = clf.coef_
    return W

def feature_ranking(W):
    """
    This function computes MCFS score and ranking features according to feature weights matrix W
    """
    W = abs(W)
    mcfs_score = W.max(1)
    idx = np.argsort(mcfs_score, 0)
    idx = idx[::-1]
    return idx

def eval_cluster_prediction(x_selected, y_label, n_clusters):
    """
    This function is the evaluation method of mcfs's result.

    :param x_selected:  {numpy array}, shape (n_samples, n_select)
        data matrix after feature selection
    :param y_label: {numpy array}, shape (n_samples)
        true label
    :param n_clusters: {int}
        number of clusters
    :return: nmi: {float}
    """
    k_means = KMeans(n_clusters=n_clusters)

    k_means.fit(x_selected)
    y_predict = k_means.labels_
    n_sample, n_feature = x_selected.shape
    y_label = y_label.reshape(n_sample,)

    # calculate NMI
    nmi = normalized_mutual_info_score(y_label, y_predict)

    return nmi
