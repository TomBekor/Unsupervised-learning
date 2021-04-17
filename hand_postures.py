from Data import *
from StatisticalTest import *
import matplotlib.pyplot as plt


def main():
    path = "HandPostures/DataSet/allUsers.lcl.csv"
    sample = 15000
    dim_reduction_to_visualize = 'TSNE'
    read_from_csv = True

    data = Data()
    data.preprocess(name='HandPostures', path=path, sample=sample)

    '''
    # When running for the first time, set read_from_csv to False.
    # The program will save the dimension reduction results to a csv file.
    # Then, on your second run, you can set read_from_csv to True, so the dimension reduction
    # process will be faster, and the reduced data will be loaded from the previous results.
    '''
    data.dimension_reduction(method='PCA', n_components=0.80, visualize=False,
                             read_from_csv=read_from_csv, write_to_csv=not read_from_csv)
    data.dimension_reduction(method=dim_reduction_to_visualize, n_components=2, visualize=True,
                             read_from_csv=read_from_csv, write_to_csv=not read_from_csv)

    # --------------- Silhouette score statistical tests --------------- #

    # --------------- Silhouette scores plots: --------------- #

    plots = ['Silhouette', 'WSS']
    max_clusters = 15
    # data.optimize_method(method_name='K-Means', method=data.k_means, max_clusters=max_clusters, plots=plots)
    # data.optimize_method(method_name='FCM', method=data.fcm, max_clusters=max_clusters, plots=plots)
    # data.optimize_method(method_name='GMM', method=data.gmm, max_clusters=max_clusters, plots=plots)
    # data.optimize_method(method_name='Agglomerative Hierarchical Clustering', method=data.hierarchical, max_clusters=max_clusters, plots=plots)
    # data.optimize_method(method_name='Spectral', method=data.spectral, max_clusters=max_clusters, plots=plots)

    # --------------- Optimized Silhouette parameters --------------- #

    km_n_clusters = [1, 2]
    fcm_n_clusters = [1, 2]
    gmm_n_clusters = [4, 6]
    hir_n_clusters = [1, 2, 3, 4, 5]
    spec_n_clusters = [4, 5, 6, 7]
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
    for target in data.targets():
        clustering.append(data.target_plot(target))

    # --------------- Final optimized methods --------------- #

    # After looking at the Fowlkes Mallows scores of the optimized methods with Silhouette score,
    # these will be our final optimized clustering methods:

    optimized_km = None
    optimized_fcm = None
    optimized_gmm = None
    optimized_hir = None
    optimized_spec = None
    optimized_dbscan = None

    # optimized_km = data.k_means(n_clusters=5)
    # optimized_fcm = data.fcm(n_clusters=2)
    # optimized_gmm = data.gmm(n_clusters=2)
    # optimized_hir = data.hierarchical(n_clusters=2)
    # optimized_spec = data.spectral(n_clusters=2)
    # optimized_dbscan = data.dbscan(epsilon=50, min_samples=2)

    optimized_clustering = [optimized_km, optimized_fcm, optimized_gmm, optimized_hir, optimized_spec, optimized_dbscan]

    for method in optimized_clustering:
        if method:
            clustering.append(method)

    # --------------- plot all --------------- #

    for method in clustering:
        data.plot_clusters(method)

    plt.show()


if __name__ == "__main__":
    main()
