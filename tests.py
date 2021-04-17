

# --------------- Silhouette score statistical tests --------------- #

# # p-value satisfaction:
# significance = 0.05
#
# # null hypothesis - for each clustering method, the percentage of top silhouette scores.
#
# statistical_tests = []
# km_st = StatisticalTest('K-Means', data, data.k_means, significance, H0=0.70, sample=1000, n_tests=1000)
# fcm_st = StatisticalTest('FCM', data, data.fcm, significance, H0=0.99, sample=1000, n_tests=1000)
# gmm_st = StatisticalTest('GMM', data, data.gmm, significance, H0=0.9, sample=100, n_tests=1000)
# hir_st = StatisticalTest('Agglomerative Hierarchical Clustering', data, data.hierarchical, significance, H0=0.70, sample=100, n_tests=1000)
# spec_st = StatisticalTest('Spectral', data, data.spectral, significance, H0=0.70, sample=100, n_tests=1000)
#
# # statistical_tests.append(km_st)
# # statistical_tests.append(fcm_st)
# # statistical_tests.append(gmm_st)
# # statistical_tests.append(hir_st)
# # statistical_tests.append(spec_st)
#
# for statistical_test in statistical_tests:
#     statistical_test.calc_silhouette_scores(50)
#
# # km_st.find_maximum_silhouette(top_results=5)
# # fcm_st.find_maximum_silhouette(top_results=2)
# # gmm_st.find_maximum_silhouette(top_results=2)
# # hir_st.find_maximum_silhouette(top_results=5)
# # spec_st.find_maximum_silhouette(top_results=5)
#
# for statistical_test in statistical_tests:
#     statistical_test.perform_test()
#
# # km_st.write_results('e-shop clothing 2008/StatisticalTests/K_Means_Statistical_test_results.txt')
# # fcm_st.write_results('e-shop clothing 2008/StatisticalTests/FCM_Statistical_test_results.txt')
# # gmm_st.write_results('e-shop clothing 2008/StatisticalTests/GMM_Statistical_test_results.txt')
# # hir_st.write_results('e-shop clothing 2008/StatisticalTests/Hierarchical_Statistical_test_results.txt')
# # spec_st.write_results('e-shop clothing 2008/StatisticalTests/Spectral_Statistical_test_results.txt')