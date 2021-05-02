from typing import Callable
from Clustering import *
from sklearn.metrics import silhouette_score
from Data import Data
from scipy import stats
import os

# TODO: https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f

class StatisticalTest:
    def __init__(self, method_name: str, data: Data, method: Callable[[int], Clustering],
                 significance, sample: int, n_tests: int):
        self.method_name = method_name
        self.reduced_data = data.get_reduced_data()
        self.method = method
        self.significance = significance
        self.sample = sample
        self.n_tests = n_tests
        self.max_clusters = None
        self.clusters = None
        self.silhouette_scores = None
        self.final_t_stat = None
        self.final_p_value = None
        self.optimized_number_of_clusters = None
        self.testing_history = ['Testing history:\n']
        self.optimized_n_options = []

    def calc_silhouette_scores(self, max_clusters: int):
        self.max_clusters = max_clusters
        print('starting statistical test on', self.method_name, '...')
        silhouette = {}
        for n_clusters in range(2, self.max_clusters + 1):
            clustering = self.method(n_clusters)
            labels = clustering.get_labels()
            # will run n_tests silhouette scores for the statistical test:
            silhouette_test = []
            for test in range(self.n_tests):
                silhouette_test.append(silhouette_score(X=self.reduced_data.values.tolist(), labels=labels,
                                                        sample_size=self.sample, random_state=test))
            silhouette[n_clusters] = silhouette_test
        self.silhouette_scores = silhouette
        print('done.')

    def optimize_clusters_number(self):
        optimized_n = 2
        for n_clusters in self.silhouette_scores.keys():
            if not n_clusters == optimized_n:
                t_statistic, p_value = stats.ttest_ind(self.silhouette_scores[optimized_n],
                                                       self.silhouette_scores[n_clusters],
                                                       alternative='greater')

                null_h = 'Null-Hypothesis: clustering with %d clusters have ' \
                         'the same Silhouette score as clustering with %d clusters.\n' \
                         'Alternative-Hypothesis: clustering with %d clusters have ' \
                         'greater Silhouette score than clustering with %d clusters.\n' % \
                         (optimized_n, n_clusters, optimized_n, n_clusters)

                results = 'Results: t-stat = %0.3f, p-value = %0.3f.\n' % (t_statistic, p_value)

                conclusion = 'The Null-Hypothesis rejected and it proves that ' \
                             'the Alternative-Hypothesis is correct and statistically significant.\n\n\n'

                if p_value >= self.significance:
                    conclusion = 'Failed to reject the Null-Hypothesis.\n\n\n'

                if p_value >= 1 - self.significance:
                    self.final_t_stat, self.final_p_value = stats.ttest_ind(self.silhouette_scores[n_clusters],
                                                                            self.silhouette_scores[optimized_n],
                                                                            alternative='greater')
                    optimized_n = n_clusters


                test = null_h + results + conclusion

                self.testing_history.append(test)

        if not self.final_p_value:
            t_statistic, p_value = stats.ttest_ind(self.silhouette_scores[2],
                                                   self.silhouette_scores[10],
                                                   alternative='greater')
            if p_value <= self.significance:
                self.final_t_stat = t_statistic
                self.final_p_value = p_value

        # check for same level silhouette score on clusters number:
        self.optimized_n_options.append(optimized_n)
        for n_clusters in self.silhouette_scores.keys():
            if not n_clusters == optimized_n:
                t_statistic, p_value = stats.ttest_ind(self.silhouette_scores[optimized_n],
                                                       self.silhouette_scores[n_clusters],
                                                       alternative='greater')
                if self.significance < p_value < 1 - self.significance:
                    self.optimized_n_options.append(n_clusters)

        self.optimized_number_of_clusters = optimized_n

    def write_results(self, results_dir):
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        results_file_name = results_dir + self.method_name
        with open(results_file_name, 'w+') as results_file:
            results_file.write(self.method_name + ' Statistical Test Results:\n\n')

            if len(self.optimized_n_options) == 1:
                results_file.write('The optimal number of clusters is: ' + str(self.optimized_number_of_clusters) + ', ')
                results_file.write('with t-stat of: %0.3f, and p-value of: %0.3f.' % (self.final_t_stat, self.final_p_value))
            else:
                results_file.write('The optimal number of clusters are: ')
                results_file.write(str(self.optimized_n_options)[1:-1])

            results_file.write('\n\n')

            for test in self.testing_history:
                results_file.write(test)
                results_file.write('\n')
        results_file.close()

