from Data import *
from StatisticalTest import *
import matplotlib.pyplot as plt


def main():
    name = 'Pulsar Stars'
    subject_path = name.replace(' ', '') + '/'
    data_path = subject_path + "DataSet/HTRU_2.csv"
    sample = 15000

    dim_reduction_to_visualize = 'PCA'
    read_from_csv = False
    anomaly_detection = False
    figsize = (7, 5)

    perform_statistical_tests = False
    compute_fowlkes_mallows = False
    perform_clustering = True

    data_dir = subject_path + 'with_anomaly_detection/' if anomaly_detection else subject_path + 'without_anomaly_detection/'
    plots_dir = data_dir + 'Plots/'

    data = Data()
    data.preprocess(name='PulsarStars', path=data_path, sample=sample, anomaly_detection=anomaly_detection,
                    data_dir=data_dir, plots_dir=plots_dir, figsize=figsize)

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

    # p-value satisfaction:
    significance = 0.05

    # null hypothesis in each t-test - for each clustering method, finds the optimal number of clusters using silhouette score.

    statistical_tests = []
    km_st = StatisticalTest('K-Means', data, data.k_means, significance, sample=1000, n_tests=50)
    fcm_st = StatisticalTest('FCM', data, data.fcm, significance, sample=1000, n_tests=50)
    gmm_st = StatisticalTest('GMM', data, data.gmm, significance, sample=1000, n_tests=50)
    hir_st = StatisticalTest('Agglomerative Hierarchical Clustering', data, data.hierarchical, significance, sample=1000, n_tests=50)
    spec_st = StatisticalTest('Spectral', data, data.spectral, significance, sample=1000, n_tests=50)

    if perform_statistical_tests:
        statistical_tests.append(km_st)
        statistical_tests.append(fcm_st)
        statistical_tests.append(gmm_st)
        statistical_tests.append(hir_st)
        statistical_tests.append(spec_st)

    for statistical_test in statistical_tests:
        statistical_test.calc_silhouette_scores(max_clusters=10)
        statistical_test.optimize_clusters_number()
        statistical_test.write_results(data_dir + 'StatisticalTests/')

    if perform_statistical_tests:
        if anomaly_detection:
            anomaly_title = ' with anomaly detection'
        else:
            anomaly_title = ' without anomaly detection'
        tests_title = name + anomaly_title
        plot_statistical_tests(statistical_tests, tests_title, plots_dir, figsize=figsize, legend=True)

    # --------------- Silhouette scores plots: --------------- #

    plots = ['Silhouette', 'WSS']
    max_clusters = 10
    # data.optimize_method(method_name='K-Means', method=data.k_means, max_clusters=max_clusters, plots=plots)
    # data.optimize_method(method_name='FCM', method=data.fcm, max_clusters=max_clusters, plots=plots)
    # data.optimize_method(method_name='GMM', method=data.gmm, max_clusters=max_clusters, plots=plots)
    # data.optimize_method(method_name='Agglomerative Hierarchical Clustering', method=data.hierarchical, max_clusters=max_clusters, plots=plots)
    # data.optimize_method(method_name='Spectral', method=data.spectral, max_clusters=max_clusters, plots=plots)

    # --------------- Optimized Silhouette parameters --------------- #

    if not anomaly_detection:  # without anomaly detection
        km_n_clusters = [2]
        fcm_n_clusters = [2]
        gmm_n_clusters = [2]
        hir_n_clusters = [2]
        spec_n_clusters = [2]
        dbscan_params = [np.linspace(0.5, 5, 10),  # epsilons.
                         np.linspace(2, 11, 10)]  # min samples - have to be integers.

    else:  # with anomaly detection
        km_n_clusters = [3]
        fcm_n_clusters = [2, 3]
        gmm_n_clusters = [3]
        hir_n_clusters = [3]
        spec_n_clusters = [3]
        dbscan_params = [np.linspace(0.5, 5, 10),  # epsilons.
                         np.linspace(2, 11, 10)]  # min samples - have to be integers.

    # --------------- Fowlkes Mallows scores on the optimized methods with Silhouette score --------------- #

    data.set_palette_color('viridis')

    if compute_fowlkes_mallows:
        apply_method(data, data.k_means, km_n_clusters)
        apply_method(data, data.fcm, fcm_n_clusters)
        apply_method(data, data.gmm, gmm_n_clusters)
        apply_method(data, data.hierarchical, hir_n_clusters)
        apply_method(data, data.spectral, spec_n_clusters)
        # # apply_dbscan(data.dbscan, dbscan_params)

    # --------------- True labels plot --------------- #

    data.set_palette_color('bright')
    clustering = []
    for target in data.targets():
        clustering.append(data.target_plot(target, anomalies=True))

    # --------------- Final optimized methods --------------- #

    # After looking at the Fowlkes Mallows scores of the optimized methods with Silhouette score,
    # these will be our final optimized clustering methods:

    optimized_km = None
    optimized_fcm = None
    optimized_gmm = None
    optimized_hir = None
    optimized_spec = None
    optimized_dbscan = None

    if perform_clustering:

        if not anomaly_detection:  # without anomaly detection.
            optimized_km = data.k_means(n_clusters=2)
            optimized_fcm = data.fcm(n_clusters=2)
            optimized_gmm = data.gmm(n_clusters=2)
            optimized_hir = data.hierarchical(n_clusters=2)
            optimized_spec = data.spectral(n_clusters=2)
            # optimized_dbscan = data.dbscan(epsilon=70, min_samples=8)

        else:  # with anomaly detection.
            optimized_km = data.k_means(n_clusters=3)
            optimized_fcm = data.fcm(n_clusters=2)
            optimized_gmm = data.gmm(n_clusters=3)
            optimized_hir = data.hierarchical(n_clusters=3)
            optimized_spec = data.spectral(n_clusters=3)
            # optimized_dbscan = data.dbscan(epsilon=70, min_samples=8)

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
