import pandas as pd
import numpy as np
from typing import Callable
from Clustering import *
from sklearn.metrics import silhouette_score
from Data import Data
from statsmodels.stats.proportion import proportions_ztest


class StatisticalTest:
    def __init__(self, method_name: str, data: Data, method: Callable[[int], Clustering],
                 significance, H0, sample: int, n_tests: int):
        self.method_name = method_name
        self.reduced_data = data.get_reduced_data()
        self.method = method
        self.significance = significance
        self.H0 = H0
        self.sample = sample
        self.n_tests = n_tests
        self.clusters = None
        self.silhouette_scores = None
        self.results = None
        self.best_n = None
        self.appearances = None
        self.top_results = None
        self.z_stat = None
        self.p_value = None

    def calc_silhouette_scores(self, max_clusters: int):
        print('starting statistical test on', self.method_name, '...')
        silhouette = []
        for n_clusters in range(2, max_clusters + 1):
            clustering = self.method(n_clusters)
            labels = clustering.get_labels()
            # will run n_tests silhouette scores for the statistical test:
            silhouette_test = []
            for test in range(self.n_tests):
                silhouette_test.append(silhouette_score(X=self.reduced_data.values.tolist(), labels=labels,
                                                        sample_size=self.sample, random_state=test))
            silhouette.append(silhouette_test)
        self.silhouette_scores = silhouette
        print('done.')

    def find_maximum_silhouette(self, top_results: int):
        numpy_array = np.array(self.silhouette_scores)
        transpose = numpy_array.T
        self.silhouette_scores = transpose.tolist()
        for test in range(len(self.silhouette_scores)):
            self.silhouette_scores[test] = np.argmax(self.silhouette_scores[test]) + 2
        results = pd.DataFrame(self.silhouette_scores)[0]
        self.results = results.value_counts()
        self.best_n = []
        self.appearances = 0
        self.top_results = min(top_results, len(self.results.index))
        for i in range(self.top_results):
            self.best_n.append(self.results.index[i])
            self.appearances += self.results.values[i]
        return self.results

    def perform_test(self):
        sample_success = self.appearances
        sample_size = self.n_tests
        self.z_stat, self.p_value = proportions_ztest(count=sample_success, nobs=sample_size,
                                                      value=self.H0, alternative='larger')

    def write_results(self, results_file_name):
        with open(results_file_name, 'w+') as results_file:
            results_file.write(self.method_name + ' Statistical Test Results\n\n')
            results_file.write('the top ' + str(self.top_results) + ' optimal number of clusters are: ' +
                               str(self.best_n)[1: -1] + '\n')
            results_file.write('\n')
            results_file.write('times success: ' + str(self.appearances) + '\n')
            results_file.write('times searched: ' + str(self.n_tests) + '\n')
            results_file.write('\n')
            results_file.write('z-stat: %0.3f, p-value: %0.3f' % (self.z_stat, self.p_value))
            results_file.write('\n\n')
            if self.p_value < self.significance:
                results_file.write('Reject the null hypothesis - the alternative hypothesis is true.\n')
                results_file.write('More than ' + str(self.H0 * 100) + '% of the silhouette score samples found the same top '
                                   + str(self.top_results) + ' number of clusters.')
            else:
                results_file.write('Fail to reject the null hypothesis - we have nothing else to say')
            results_file.write('\n\n')
            results_file.write('full results:\n')
            results_file.write(str(self.results))
