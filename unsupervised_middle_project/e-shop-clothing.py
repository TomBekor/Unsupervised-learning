from Data import *
from StatisticalTest import *
import matplotlib.pyplot as plt


def main():
    path = "e-shop clothing 2008/DataSet/e-shop data and description/e-shop clothing 2008.csv"
    sample = 15000
    dim_reduction_to_visualize = 'TSNE'
    read_from_csv = True

    data = Data()
    data.preprocess(path=path, sample=sample)

    # TODO delete the read from csv and write to csv.
    data.dimension_reduction(method='PCA', n_components=0.95, visualize=False,
                             read_from_csv=read_from_csv, write_to_csv=not read_from_csv)
    data.dimension_reduction(method=dim_reduction_to_visualize, n_components=2, visualize=True,
                             read_from_csv=read_from_csv, write_to_csv=not read_from_csv)

    # --------------- Silhouette score statistical tests --------------- #

    # p-value satisfaction:
    significance = 0.05

    # null hypothesis - for each clustering method, the percentage of top silhouette scores.

    statistical_tests = []
    km_st = StatisticalTest('K-Means', data, data.k_means, significance, H0=0.70, sample=1000, n_tests=1000)
    fcm_st = StatisticalTest('FCM', data, data.fcm, significance, H0=0.99, sample=1000, n_tests=1000)
    gmm_st = StatisticalTest('GMM', data, data.gmm, significance, H0=0.9, sample=100, n_tests=1000)
    hir_st = StatisticalTest('Agglomerative Hierarchical Clustering', data, data.hierarchical, significance, H0=0.70, sample=100, n_tests=1000)
    spec_st = StatisticalTest('Spectral', data, data.spectral, significance, H0=0.70, sample=100, n_tests=1000)

    # statistical_tests.append(km_st)
    # statistical_tests.append(fcm_st)
    # statistical_tests.append(gmm_st)
    # statistical_tests.append(hir_st)
    # statistical_tests.append(spec_st)

    for statistical_test in statistical_tests:
        statistical_test.calc_silhouette_scores(50)

    # km_st.find_maximum_silhouette(top_results=5)
    # fcm_st.find_maximum_silhouette(top_results=2)
    # gmm_st.find_maximum_silhouette(top_results=2)
    # hir_st.find_maximum_silhouette(top_results=5)
    # spec_st.find_maximum_silhouette(top_results=5)

    for statistical_test in statistical_tests:
        statistical_test.perform_test()

    # km_st.write_results('e-shop clothing 2008/StatisticalTests/K_Means_Statistical_test_results.txt')
    # fcm_st.write_results('e-shop clothing 2008/StatisticalTests/FCM_Statistical_test_results.txt')
    # gmm_st.write_results('e-shop clothing 2008/StatisticalTests/GMM_Statistical_test_results.txt')
    # hir_st.write_results('e-shop clothing 2008/StatisticalTests/Hierarchical_Statistical_test_results.txt')
    # spec_st.write_results('e-shop clothing 2008/StatisticalTests/Spectral_Statistical_test_results.txt')

    # --------------- Silhouette scores plots: --------------- #

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
    for target in data.targets():
        clustering.append(data.target_plot(target))

    # --------------- Final optimized methods --------------- #

    # After looking at the Fowlkes Mallows scores of the optimized methods with Silhouette score,
    # these will be our final optimized clustering methods:

    # optimized_km = data.k_means(n_clusters=19)
    # optimized_fcm = data.fcm(n_clusters=2)
    # optimized_gmm = data.gmm(n_clusters=2)
    # optimized_hir = data.hierarchical(n_clusters=20)
    # optimized_spec = data.spectral(n_clusters=2)
    # optimized_dbscan = data.dbscan(epsilon=2.5, min_samples=2)

    # clustering.append(optimized_km)
    # clustering.append(optimized_fcm)
    # clustering.append(optimized_gmm)
    # clustering.append(optimized_hir)
    # clustering.append(optimized_spec)
    # clustering.append(optimized_dbscan)

    # --------------- plot all --------------- #

    for clustering in clustering:
        data.plot_clusters(clustering)

    plt.show()


if __name__ == "__main__":
    main()
