import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from typing import Callable
import seaborn as sns


class Data:
    def __init__(self, path: str):
        self.name = path.split(sep='/')[-1][:-4]
        self.data = pd.read_csv(path, delimiter=";").sample(n=10000, random_state=100)
        self.reduced_data = None
        self.colors = ['b']

    def preprocess(self):
        if self.name == 'data #1':
            # TODO first data set preprocess
            pass
        elif self.name == 'data #2':
            # TODO second data set preprocess
            pass
        elif self.name == 'e-shop clothing 2008':
            self.data = self.data.drop(['year', 'page 2 (clothing model)'], axis=1)

    def normalization(self):
        scaler = StandardScaler()
        self.data = pd.DataFrame(data=scaler.fit_transform(self.data), columns=[self.data.columns])

    def dimension_reduction(self, method: str, n_components: int, clustering_on_reduced_data: bool,
                            read_from_csv: bool = False, write_to_csv: bool = False):
        if read_from_csv:
            self.reduced_data = pd.read_csv(method + '_reduced_data.csv')
        else:
            if method == 'PCA':
                print('Starting dimension reduction using PCA...', end=' ')
                pca = PCA(n_components=n_components)
                principal_components = pca.fit_transform(self.data)
                self.reduced_data = pd.DataFrame(data=principal_components, columns=['dim 1', 'dim 2'])
                print('done')
                print('pca components variance ratio:')
                print(pca.explained_variance_ratio_)
                print('total variance:', str(int((sum(pca.explained_variance_ratio_) * 100))) + '%')
            elif method == 'TSNE':
                print('Starting dimension reduction using TSNE...', end=' ')
                tsne = TSNE(n_components=n_components)
                embedded_data = tsne.fit_transform(self.data)
                self.reduced_data = pd.DataFrame(data=embedded_data, columns=['dim 1', 'dim 2'])
                print('done')
        if clustering_on_reduced_data:
            self.data = self.reduced_data
        if write_to_csv:
            self.reduced_data.to_csv(method + '_reduced_data.csv', index=False)

    # -------------------------------- clustering -------------------------------- #

    # ------------- plot ------------- #

    def cluster_color(self, label):
        return self.colors[label % len(self.colors)]

    def plot_clusters(self, clustering, title_fs=14, label_fs=10, ticks_fs=8):
        labels = clustering[0]
        n_clusters = clustering[1]
        title = clustering[2]
        self.colors = sns.color_palette('bright', n_colors=n_clusters).as_hex()
        for label in range(n_clusters):
            plt.scatter(self.reduced_data.values[labels == label, 0], self.reduced_data.values[labels == label, 1],
                        s=40, c=self.cluster_color(label))
        clusters_ax = plt.gca()
        clusters_ax.set_title(title, fontsize=title_fs)
        clusters_ax.set_xlabel('dim1', fontsize=label_fs)
        clusters_ax.set_ylabel('dim2', fontsize=label_fs)
        plt.show()

    # ------------- optimization ------------- #

    def optimize_method(self, method_name: str, method: Callable[[int], list], max_clusters: int,
                        label_fs=10, ticks_fs=8):
        print('optimizing', method_name + '...')
        silhouette = []
        for n_clusters in range(2, max_clusters + 1):
            labels = method(n_clusters)[0]
            silhouette.append((n_clusters, silhouette_score(X=self.data.values.tolist(), labels=labels, sample_size=3000
                                                            , random_state=100)))
        print('done')
        # plot:
        title = method_name + ' Silhouette Score'
        silhouette_df = pd.DataFrame(data=silhouette, columns=['number of clusters', 'Silhouette Score'])
        ax = silhouette_df.plot(x='number of clusters', y='Silhouette Score')
        ax.set_title(label=title)
        ax.set_xlabel('number of clusters', fontsize=label_fs)
        ax.set_ylabel('Silhouette Score', fontsize=label_fs)
        ax.set_xticks(range(2, max_clusters, 2))
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(ticks_fs)
        plt.show()

    def plot_dendrogram(self, horizontal_cut: float = 3):
        plt.title("Dendrogram")
        dendrogram(linkage(self.data, method='ward'))
        # plt.axhline(y=horizontal_cut, color='r', linestyle='--')
        plt.show()

        # ------------- methods ------------- #

    def k_means(self, n_clusters: int):
        print('performing k means clustering with', n_clusters, 'clusters...', end=' ')
        kmeans = KMeans(n_clusters=n_clusters, algorithm='full')
        labels = kmeans.fit_predict(self.data)
        print('done')
        return labels, n_clusters, 'K-Means with ' + str(n_clusters) + ' clusters'

    def fcm(self, n_clusters: int):
        print('performing fcm clustering with', n_clusters, 'clusters...', end=' ')
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(self.data.T.values, n_clusters, 2, error=0.005, maxiter=1000)
        print('done')
        return np.argmax(u, axis=0), n_clusters, 'FCM with ' + str(n_clusters) + ' clusters'

    def gmm(self, n_clusters: int, verbose=0):
        end = ' ' if verbose == 0 else '\n'
        print('performing gmm clustering with', n_clusters, 'clusters...', end=end)
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', init_params='kmeans'
                              , warm_start=False, n_init=5, random_state=100, verbose=verbose)
        gmm.fit(self.data)
        print('done')
        return gmm.predict(self.data), n_clusters, 'GMM with ' + str(n_clusters) + ' clusters'

    def hierarchical(self, n_clusters: int, linkage: str = 'ward'):
        # TODO amount of points to use - hierarchical
        print('performing hierarchical clustering with', n_clusters, 'clusters...', end=' ')
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        hierarchical.fit(self.data)
        print('done')
        return hierarchical.labels_, n_clusters, 'Agglomerative Hierarchical Clustering with ' \
               + str(n_clusters) + ' clusters'

    def spectral(self, n_clusters: int):
        # TODO amount of points to use - spectral
        print('performing spectral clustering with', n_clusters, 'clusters...')
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=30,
                                      assign_labels='kmeans', n_init=5)
        spectral.fit(self.data)
        print('done')
        return spectral.labels_, n_clusters, 'Spectral Clustering with ' + str(n_clusters) + ' clusters'
