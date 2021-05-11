from typing import Callable, List
from Clustering import *
from sklearn.metrics import silhouette_score
from Data import Data
from scipy import stats
import os
import matplotlib.pyplot as plt


# https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f

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
        self.silhouette_means = {}
        self.final_t_stat = None
        self.final_p_value = None
        self.optimized_number_of_clusters = None
        self.testing_history = ['Testing history:\n']
        self.optimized_n_options = []
        self.anova_t_stat = None
        self.anova_p_value = None
        self.anova_history = None

    def get_method_name(self):
        return self.method_name

    def get_means(self):
        return list(range(2, self.max_clusters + 1)), self.silhouette_means

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
        for key in self.silhouette_scores.keys():
            self.silhouette_means[key] = sum(self.silhouette_scores[key]) / len(self.silhouette_scores[key])
        print('done.')

    def optimize_clusters_number(self):

        self.anova_t_stat, self.anova_p_value = stats.f_oneway(*list(self.silhouette_scores.values()))

        null_h = 'Null-Hypothesis: clustering with specific number of clusters ' \
                 'have no affect, i.e. they all have the same Silhouette score.\n' \
                 'Alternative-Hypothesis: They all have different Silhouette score.\n'

        results = 'Results: t-stat = %0.3f, p-value = %0.3f.\n' % (self.anova_t_stat, self.anova_p_value)

        conclusion = 'The Null-Hypothesis rejected and it proves that ' \
                     'the Alternative-Hypothesis is correct and statistically significant.\n\n\n'

        if self.anova_p_value >= 0.05:
            conclusion = 'Failed to reject the Null-Hypothesis.\n\n\n'

        test = null_h + results + conclusion

        self.anova_history = test

        if self.anova_p_value < 0.5:

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

            results_file.write(self.anova_history)

            results_file.write('Silhouette means:\n')
            for n_clusters in self.silhouette_means:
                results_file.write(str(n_clusters) + ' clusters Silhouette Score: ' + str(self.silhouette_means[n_clusters])[:5] + '\n')

            results_file.write('\n\n')
            if self.anova_p_value < 0.5:
                if len(self.optimized_n_options) == 1:
                    results_file.write(
                        'The optimal number of clusters is: ' + str(self.optimized_number_of_clusters) + ', ')
                    results_file.write(
                        'with t-stat of: %0.3f, and p-value of: %0.3f.' % (self.final_t_stat, self.final_p_value))
                else:
                    results_file.write('The optimal number of clusters are: ')
                    results_file.write(str(self.optimized_n_options)[1:-1])

                results_file.write('\n\n')

                for test in self.testing_history:
                    results_file.write(test)
                    results_file.write('\n')
        results_file.close()


def plot_statistical_tests(tests: List[StatisticalTest], tests_title, plots_dir,
                           title_fs=14, label_fs=12, ticks_fs=12, figsize=(7, 5), legend=True):
    plt.figure(figsize=figsize)
    title = tests_title
    x_label = 'number of clusters'
    y_label = 'mean Silhouette score'

    for test in tests:
        method = test.get_method_name()
        test_clusters, test_silhouette = test.get_means()
        plt.plot(test_clusters, test_silhouette.values(), label=method, marker='o')

    ax = plt.gca()
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel(x_label, fontsize=label_fs)
    ax.set_ylabel(y_label, fontsize=label_fs)
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ticks_fs)

    if legend:
        plt.legend(prop={'size': ticks_fs})
    plt.savefig(str(plots_dir + title + ' statistical tests plot').replace(' ', '_'))






