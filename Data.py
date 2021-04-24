from typing import Callable
import pandas as pd
import numpy as np
import seaborn as sns
from Clustering import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import fowlkes_mallows_score
# import skfuzzy as fuzz
import preprocess
import os


class Data:
    def __init__(self):
        self.name = None
        self.data = None
        self.target = None
        self.data = None
        self.reduced_data = None
        self.visualized_data = None
        self.colors = ['b']
        self.palette = 'bright'

    def targets(self):
        return self.target.columns.tolist()

    def get_reduced_data(self):
        return self.reduced_data

    def set_palette_color(self, palette: str):
        self.palette = palette

    def preprocess(self, name: str, path: str, sample: int):
        self.name = name
        if self.name == 'HandPostures':
            self.data, self.target = preprocess.hand_postures_preprocess(path, sample)
        elif self.name == 'diabetic_data':
            # TODO second data.
            pass

    def dimension_reduction(self, method: str, n_components: float, visualize: bool,
                            read_from_csv: bool = False, write_to_csv: bool = False):
        if not os.path.exists(self.name + '/DataCsv'):
            os.mkdir(self.name + '/DataCsv')
        if visualize:
            file_name = self.name + '/DataCsv/' + method + '_' + 'visualized' + '_data.csv'
        else:
            file_name = self.name + '/DataCsv/' + method + '_' + 'reduced' + '_data.csv'
        if read_from_csv:
            result = pd.read_csv(file_name)
        else:
            if method == 'PCA':
                print('Starting dimension reduction using PCA...', end=' ')
                pca = PCA(n_components=n_components, svd_solver='full')
                principal_components = pca.fit_transform(self.data)
                result = pd.DataFrame(data=principal_components)
                print('done')
                print('from', len(self.data.columns), 'dimensions to', pca.n_components_, 'pca components, variance ratio:')
                print(pca.explained_variance_ratio_)
                print('total variance:', str(int((sum(pca.explained_variance_ratio_) * 100))) + '%')
            elif method == 'TSNE':
                print('Starting dimension reduction using TSNE...', end=' ')
                tsne = TSNE(n_components=n_components)
                embedded_data = tsne.fit_transform(self.reduced_data)
                result = pd.DataFrame(data=embedded_data, columns=['dim 1', 'dim 2'])
                print('done')
            else:
                # TODO ready for another dimension reduction.
                result = None
        if visualize:
            self.visualized_data = result
        else:
            self.reduced_data = result
        if write_to_csv:
            result.to_csv(file_name, index=False)

    # -------------------------------- clustering -------------------------------- #

    def target_plot(self, label: str):
        labels = self.target[label].values
        n_labels = max(pd.unique(labels).max(), len(pd.unique(labels))) + 1
        return Clustering(labels=labels, n_clusters=n_labels, title=label.capitalize() + ' True Labels', palette=self.palette)

    # ------------- plot ------------- #

    def cluster_color(self, label):
        if int(label) == -1:
            return 'k'
        return self.colors[int(label) % len(self.colors)]

    def plot_clusters(self, clustering: Clustering, title_fs=14, label_fs=10, ticks_fs=8):
        plt.figure()
        labels = clustering.get_labels()
        n_clusters = clustering.get_n_clusters()
        title = clustering.get_title()
        self.colors = sns.color_palette(palette=clustering.get_palette(), n_colors=n_clusters).as_hex()
        plt.scatter(self.visualized_data.values[:, 0], self.visualized_data.values[:, 1],
                    s=40, c=[self.cluster_color(label) for label in labels])
        clusters_ax = plt.gca()
        clusters_ax.set_title(title, fontsize=title_fs)
        clusters_ax.set_xlabel('dim1', fontsize=label_fs)
        clusters_ax.set_ylabel('dim2', fontsize=label_fs)

    # ------------- optimization ------------- #

    def optimize_method(self, method_name: str, method: Callable[[int], Clustering], max_clusters: int,
                        label_fs=10, ticks_fs=8, plots=None):
        if plots is None:
            plots = ['Silhouette', 'Fowlkes Mallows', 'WSS']
        # will create wss elbow in case of k-means
        print('optimizing', method_name + '...')
        clusters = []
        silhouette = []
        fms = {}  # fowlkes mallows score
        for cat in self.target.columns:
            fms[cat] = []
        wss = []
        for n_clusters in range(2, max_clusters + 1):
            clustering = method(n_clusters)
            labels = clustering.get_labels()
            clusters.append(n_clusters)

            # calculate silhouette score with random_state == 100 and 5000 samples:
            silhouette.append(silhouette_score(X=self.reduced_data.values.tolist(), labels=labels,
                                               sample_size=5000, random_state=100))

            # calculate fowlkes_mallows_score:
            for cat in self.target.columns:
                true_labels = self.target[cat].values
                fms[cat].append(fowlkes_mallows_score(labels_true=true_labels, labels_pred=labels))

            # calculate wss:
            if method_name == 'K-Means':
                wss.append(clustering.get_inertia())

        # plot optimization:
        def plot_opt(y, opt_title: str, y_label: str):
            plt.figure()
            plt.plot(clusters, y, '-o')
            ax = plt.gca()
            ax.set_title(label=opt_title)
            ax.set_xlabel('number of clusters', fontsize=label_fs)
            ax.set_ylabel(y_label, fontsize=label_fs)
            ax.set_xticks(range(2, max_clusters, 2))
            for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(ticks_fs)

        # plot silhouette:
        if 'Silhouette' in plots:
            title = method_name + ' Silhouette Score'
            plot_opt(y=silhouette, opt_title=title, y_label='Silhouette Score')

        # plot fowlkes mallows:
        if 'Fowlkes Mallows' in plots:
            for cat in self.target.columns:
                title = method_name + ' & ' + cat + ' Fowlkes Mallows Score'
                plot_opt(y=fms[cat], opt_title=title, y_label='Fowlkes Mallows Score')

        # plot elbow method:
        if 'WSS' in plots:
            if method_name == 'K-Means':
                title = method_name + ' WSS'
                plot_opt(y=wss, opt_title=title, y_label='WSS Score')

        print('done')

    def plot_dendrogram(self, horizontal_cut: float = 3):
        plt.title("Dendrogram")
        dendrogram(linkage(self.reduced_data, method='ward'))
        plt.axhline(y=horizontal_cut, color='r', linestyle='--')

    def target_fowlkes_mallows(self, labels):
        fowlkes_mallows = {}
        for cat in self.target.columns:
            fowlkes_mallows[cat] = fowlkes_mallows_score(labels_true=self.target[cat].values, labels_pred=labels)
        return fowlkes_mallows

        # ------------- methods ------------- #

    def k_means(self, n_clusters: int) -> Clustering:
        print('performing k means clustering with', n_clusters, 'clusters...', end=' ')
        kmeans = KMeans(n_clusters=n_clusters, algorithm='full', random_state=100)
        labels = kmeans.fit_predict(self.reduced_data)
        fowlkes_mallows = self.target_fowlkes_mallows(labels)
        print('done')
        return Clustering(labels=labels, n_clusters=n_clusters, title='K-Means with ' + str(n_clusters) + ' clusters',
                          inertia=kmeans.inertia_, palette=self.palette, fowlkes_mallows=fowlkes_mallows)

    # TODO make fuzzy works, don't forget to import
    # def fcm(self, n_clusters: int) -> Clustering:
    #     print('performing fcm clustering with', n_clusters, 'clusters...', end=' ')
    #     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(self.reduced_data.T.values, n_clusters, 2, error=0.005, maxiter=1000,
    #                                                      seed=100)
    #     labels = np.argmax(u, axis=0)
    #     fowlkes_mallows = self.target_fowlkes_mallows(labels)
    #     print('done')
    #     return Clustering(labels=labels, n_clusters=n_clusters, title='FCM with ' + str(n_clusters) + ' clusters',
    #                       palette=self.palette, fowlkes_mallows=fowlkes_mallows)

    def gmm(self, n_clusters: int, verbose=0) -> Clustering:
        end = ' ' if verbose == 0 else '\n'
        print('performing gmm clustering with', n_clusters, 'clusters...', end=end)
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', init_params='kmeans'
                              , warm_start=False, n_init=5, random_state=100, verbose=verbose)
        gmm.fit(self.reduced_data)
        labels = gmm.predict(self.reduced_data)
        fowlkes_mallows = self.target_fowlkes_mallows(labels)
        print('done')
        return Clustering(labels=labels, n_clusters=n_clusters, title='GMM with ' + str(n_clusters) + ' clusters',
                          palette=self.palette, fowlkes_mallows=fowlkes_mallows)

    def hierarchical(self, n_clusters: int, linkage: str = 'ward') -> Clustering:
        # TODO amount of points to use - hierarchical
        print('performing hierarchical clustering with', n_clusters, 'clusters...', end=' ')
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        hierarchical.fit(self.reduced_data)
        labels = hierarchical.labels_
        fowlkes_mallows = self.target_fowlkes_mallows(labels)
        print('done')
        return Clustering(labels=labels, n_clusters=n_clusters, title='Agglomerative-Hierarchical clustering with ' \
                                                                      + str(n_clusters) + ' clusters',
                          palette=self.palette, fowlkes_mallows=fowlkes_mallows)

    def spectral(self, n_clusters: int) -> Clustering:
        # TODO amount of points to use - spectral
        print('performing spectral clustering with', n_clusters, 'clusters...', end=' ')
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=30,
                                      assign_labels='kmeans', n_init=5, random_state=100)
        spectral.fit(self.reduced_data)
        labels = spectral.labels_
        fowlkes_mallows = self.target_fowlkes_mallows(labels)
        print('done')
        return Clustering(labels=labels, n_clusters=n_clusters, title='Spectral Clustering with ' + str(n_clusters) + ' clusters',
                          palette=self.palette, fowlkes_mallows=fowlkes_mallows)

    def dbscan(self, epsilon: float = 0.5, min_samples: int = 5) -> Clustering:
        print('performing dbscan with epsilon=' + str(epsilon) + ' and min_samples=' + str(min_samples) + '...', end=' ')
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples).fit(self.reduced_data)
        labels = dbscan.labels_
        n_clusters = np.amax(labels) + 1
        fowlkes_mallows = self.target_fowlkes_mallows(labels)
        print('done')
        return Clustering(labels=labels, n_clusters=n_clusters, title='DBSCAN with ' + str(n_clusters) + ' clusters.' + \
                                                                      ' epsilon=' + str(epsilon) + '. min samples=' + \
                                                                      str(min_samples) + '.', palette=self.palette,
                          fowlkes_mallows=fowlkes_mallows)


def apply_method(method: Callable[[int], Clustering], n_clusters: list):
    clustering = []
    for n in n_clusters:
        clustering.append(method(n))
    fowlkes_mallows_plot(clustering, n_clusters, x_title='number of clusters')
    return clustering


def fowlkes_mallows_plot(clustering, labels, x_title, x_rotation=0):
    title = clustering[0].get_title().split(' ')[0] + ' clustering Fowlkes Mallows Scores'
    all_fowlkes_mallows = {}
    for clust in clustering:
        clust_fowlkes_mallows = clust.get_fowlkes_mallows()
        for cat in clust_fowlkes_mallows:
            if cat not in all_fowlkes_mallows:
                all_fowlkes_mallows[cat] = [clust_fowlkes_mallows[cat]]
            else:
                all_fowlkes_mallows[cat].append(clust_fowlkes_mallows[cat])
    fowlkes_mallows = pd.DataFrame(all_fowlkes_mallows)
    ax = fowlkes_mallows.plot.bar(title=title)
    ax.set_xlabel(x_title)
    ax.set_ylabel('Fowlkes Mallows Score')
    ax.set_xticklabels(labels, rotation=x_rotation)


def apply_dbscan(method: Callable[[float, int], Clustering], dbscan_params):
    dbscans = []
    for epsilon in dbscan_params[0]:
        for min_samples in dbscan_params[1]:
            dbscans.append(method(epsilon, min_samples))
    # fowlkes_mallows_plot(dbscans, dbscan_params, x_title='(epsilon,min_samples)')
    dbscan_fowlkes_mallows(dbscans, dbscan_params)
    return dbscans


def dbscan_fowlkes_mallows(dbscans, dbscan_params):
    title = 'DBSCAN clustering Fowlkes Mallows Scores'
    all_fowlkes_mallows = {}
    for clust in dbscans:
        clust_fowlkes_mallows = clust.get_fowlkes_mallows()
        for cat in clust_fowlkes_mallows:
            if cat not in all_fowlkes_mallows:
                all_fowlkes_mallows[cat] = [clust_fowlkes_mallows[cat]]
            else:
                all_fowlkes_mallows[cat].append(clust_fowlkes_mallows[cat])
    for cat in all_fowlkes_mallows:
        scores = np.array(all_fowlkes_mallows[cat])
        shape = (len(dbscan_params[0]), len(dbscan_params[1]))
        scores = scores.reshape(shape)
        fowlkes_mallows = pd.DataFrame(scores)
        fowlkes_mallows.columns = dbscan_params[1]
        fowlkes_mallows = pd.concat([fowlkes_mallows, pd.DataFrame(dbscan_params[0], columns=['min_samples'])], axis=1)
        fowlkes_mallows = fowlkes_mallows.set_index('min_samples')
        sns.heatmap(fowlkes_mallows, cmap='rocket_r')
