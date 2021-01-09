from Data import Data


def main():
    path = "e-shop data and description/e-shop clothing 2008.csv"
    data = Data(path=path)

    data.preprocess()
    data.normalization()
    # TODO delete the read from csv and write to csv.
    data.dimension_reduction(method='TSNE', n_components=2, clustering_on_reduced_data=True,
                             read_from_csv=True, write_to_csv=False)

    # data.optimize_method(method_name='K-Means', method=data.k_means, max_clusters=50)
    # data.optimize_method(method_name='FCM', method=data.fcm, max_clusters=50)
    # data.optimize_method(method_name='GMM', method=data.gmm, max_clusters=50)
    # data.optimize_method(method_name='Agglomerative Hierarchical Clustering', method=data.hierarchical, max_clusters=50)
    # data.optimize_method(method_name='Spectral', method=data.spectral, max_clusters=50)

    km = data.k_means(n_clusters=25)
    fcm = data.fcm(n_clusters=16)
    gmm = data.gmm(n_clusters=25, verbose=2)
    hir = data.hierarchical(n_clusters=25)
    spec = data.spectral(n_clusters=43)
    #
    data.plot_clusters(clustering=km)
    data.plot_clusters(clustering=fcm)
    data.plot_clusters(clustering=gmm)
    data.plot_clusters(clustering=hir)
    data.plot_clusters(clustering=spec)


if __name__ == "__main__":
    main()
