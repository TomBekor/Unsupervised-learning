from Data import *
import matplotlib.pyplot as plt


def main():
    path = "e-shop data and description/e-shop clothing 2008.csv"
    sample = 10000
    targets = ['country']
    dim_reduction_to_visualize = 'TSNE'

    data = Data(path=path, target=targets, sample=sample)
    data.preprocess()
    data.normalization()

    # TODO delete the read from csv and write to csv.
    data.dimension_reduction(method='PCA', n_components=0.95, visualize=False, read_from_csv=True, write_to_csv=False)
    data.dimension_reduction(method=dim_reduction_to_visualize, n_components=2, visualize=True, read_from_csv=True, write_to_csv=False)

    # --------------- Silhouette scores: --------------- #

    plots = ['Silhouette', 'WSS']
    # data.optimize_method(method_name='K-Means', method=data.k_means, max_clusters=50, plots=plots)
    # data.optimize_method(method_name='FCM', method=data.fcm, max_clusters=50, plots=plots)
    # data.optimize_method(method_name='GMM', method=data.gmm, max_clusters=50, plots=plots)
    # data.optimize_method(method_name='Agglomerative Hierarchical Clustering', method=data.hierarchical, max_clusters=50, plots=plots)
    # data.optimize_method(method_name='Spectral', method=data.spectral, max_clusters=50, plots=plots)

    # --------------- Optimized Silhouette parameters --------------- #

    km_n_clusters = [19, 37]
    fcm_n_clusters = [2, 3, 21, 26]
    gmm_n_clusters = [2, 4, 10]
    hir_n_clusters = [20, 26]
    spec_n_clusters = [2, 5, 15]
    dbscan_params = [np.linspace(0.5, 5, 10),  # epsilons.
                     np.linspace(2, 11, 10)]  # min samples - have to be integers.

    # --------------- Fowlkes Mallows scores on the optimized methods with Silhouette score --------------- #

    data.set_palette_color('viridis')
    # apply_method(data.k_means, km_n_clusters)
    # apply_method(data.fcm, fcm_n_clusters)
    # apply_method(data.gmm, gmm_n_clusters)
    # apply_method(data.hierarchical, hir_n_clusters)
    # apply_method(data.spectral, spec_n_clusters)
    # apply_dbscan(data.dbscan, dbscan_params)

    # --------------- True labels plot --------------- #

    data.set_palette_color('bright')
    clustering = []
    for target in targets:
        clustering.append(data.target_plot(target))

    # --------------- Final optimized methods --------------- #

    # After looking at the Fowlkes Mallows scores of the optimized methods with Silhouette score,
    # these will be our final optimized clustering methods:

    optimized_km = data.k_means(n_clusters=19)
    optimized_fcm = data.fcm(n_clusters=2)
    optimized_gmm = data.gmm(n_clusters=2)
    optimized_hir = data.hierarchical(n_clusters=20)
    optimized_spec = data.spectral(n_clusters=2)
    optimized_dbscan = data.dbscan(epsilon=2.5, min_samples=2)

    clustering.append(optimized_km)
    clustering.append(optimized_fcm)
    clustering.append(optimized_gmm)
    clustering.append(optimized_hir)
    clustering.append(optimized_spec)
    clustering.append(optimized_dbscan)

    # --------------- plot all --------------- #

    for clustering in clustering:
        data.plot_clusters(clustering)

    plt.show()


if __name__ == "__main__":
    main()
