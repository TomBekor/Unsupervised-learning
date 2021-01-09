from Data import Data


def main():
    path = "e-shop data and description/e-shop clothing 2008.csv"
    data = Data(path=path)

    data.preprocess()
    data.normalization()
    data.dimension_reduction(method='TSNE', n_components=2, clustering_on_reduced_data=True,
                             read_from_csv=True, write_to_csv=False)

    km = data.k_means(n_clusters=6)
    fcm = data.fcm(n_clusters=6)
    gmm = data.gmm(n_clusters=6)
    hir = data.hierarchical(n_clusters=6)
    spec = data.spectral(n_clusters=6)

    data.plot_clusters(clustering=km)
    data.plot_clusters(clustering=fcm)
    data.plot_clusters(clustering=gmm)
    data.plot_clusters(clustering=hir)
    data.plot_clusters(clustering=spec)


if __name__ == "__main__":
    main()
