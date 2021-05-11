# Unsupervised Learning

## Abstract

In this paper we analyzed 5 different clustering methods which were applied on
the Pulsar Stars and the Hand Postures data sets, displayed results, and analyzed
the performance for each, in attempt to find the most suiting clustering method.
We applied anomaly detection on each data set and compared the results of the
original data against the anomaly detected data. We found the Fuzzy C-Means
clustering method with 2 clusters to be the best preforming algorithm in terms of
external and internal clustering quality measures. We found the anomaly detection
was insignificant on the Hand Pulsar data set, and on the Pulsar Stars it decreased
the quality of the results.

## Running Walkthrough

To run on the Pulsar Stars data set, run the `pulsar_stars.py` python file.  
To run on the Hand Postures data set, run the `hand_postures.py` python file.

From here on those file will be called the 'main files'. 

### Configuration

At the start of each main file you can find the parameters:

* `dim_reduction_to_visualize`
* `read_from_csv`
* `anomaly_detection`
* `perform_statistical_tests`
* `compute_fowlkes_mallows`
* `perform_clustering`

Define each one as described below:  

* `dim_reduction_to_visualize` - `'PCA' \ 'TSNE'`:  
  Choose your preferred dimension reduction to visualize the data.  
  
* `read_from_csv` - `True \ False`:  
  When running for the first time(with respect to `dim_reduction_to_visualize`, set to `False`.
  On the second time, you will be able to load the dimension reduction results from the last run,
  so your run will be shorter.
  
* `anomaly_detection` - `True \ False`:  
  Choose to run with or without anomaly detection.
  
* `perform_statistical_tests` - `True \ False`:  
  Choose whether to perform statistical tests.
  
* `compute_fowlkes_mallows` - `True \ False`:  
  Choose whether to computer Fowlkes-Mallows external validation.
  
* `perform_clustering` - `True \ False`:  
  Choose whether to perform the final clustering methods.
  
These variables initialized with the recommended values for each data set.


## About

This project was made by Tom Bekor, as the final assignment for the 
Unsupervised-Learning course at Bar Ilan University, lectured by Prof. Yoram Louzoun.  
For any questions, feel free to email [tom.bekor@gmail.com](mailto:tom.bekor@gmail.com).

