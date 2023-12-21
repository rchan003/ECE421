'''
ECE421 - Assignment 1
    Implement k-means algorithm only using numPy library on the UCI ML Breast Cancer dataset 

Name: Rachel Chan
Date: September 13th, 2023 
'''
import numpy as np
import matplotlib.pyplot as plt 
import time
from sklearn.datasets import load_breast_cancer 

# Problem 1: clustering with k-means

class KMeansClustering():
    def __init__(self, n_clusters=2, min_delta=0, max_iter = 1000, normalize_X=False):
        '''
        Initialize the KMeansClustering algorithm. 

        Parameters: 
            n_cluster (int):    Number of clusters (k).  
            min_delta (float):  Min percent change in distortion before returning assignments. Between 0 & 100 inclusive with 0 = no change
            max_iter (int):     Max number of iterations to perform 

            preprocess_X (bool): If True then normalize all features of X where the maximum feature value is 1 else do nothing. 
        ''' 
        self.n_clusters = n_clusters
        self.min_delta = min_delta
        self.max_iter = max_iter
        self.normalize_X = normalize_X

    def init_clusters(self, X, random_init=False):
        '''
        Initialize the centroid locations

        Parameters:
            X (numPy array): The dataset of shape (n_samples, n_features). 
            random_init (bool): If False then pick data points from training set, else randomly pick points within the input domain.
        
        Returns: 
            clusters (numPy array): Centroid locations of shape (n_clusters, n_features). 
        '''
        if not random_init: 
            # randomly choose k points from the training set w/ no replacement (ie no repeat sampling)
            clusters = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False), :]

        else:
            # randomly choose feature value between its max & min for each centroid
            clusters = [[np.random.uniform(min(X[:,i]), max(X[:,i])) for i in range(X.shape[1])] for i in range(self.n_clusters)]
        return clusters
    
    def pre_processing(self, X):
        '''
        Normalize X features between [0,1]

        Parameters: 
            X (numPy array): The dataset of shape (n_samples, n_features).  

        Returns: 
            X_norm (numPy array): The dataset normalized such that the max value for each feature is 1
        '''
        X_norm = X / X.max(axis=0)

        return X_norm
    
    def assign_cluster(self, clusters, X):
        '''
        Assign each datapoint to its closest cluster

        Parameters: 
            clusters (numPy array): Centroid locations of shape (n_clusters, n_features). 
            X (numPy array):        The dataset of shape (n_samples, n_features)

        Returns:
            cluster_idxs (numPy array): Index of closest cluster for each sample in X. 
        '''
        cluster_idxs = [np.argmin([np.linalg.norm(x - Ci) for Ci in clusters]) for x in X]

        return np.array(cluster_idxs)
    
    def move_centroids(self, cluster_idxs, X):
        '''
        Move centroids to the mean of their assigned points

        Returns:
            new_clusters (numPy array): New centroid locations of shape (n_clusters, n_features). 
        '''
        new_clusters = np.zeros((self.n_clusters, X.shape[1]))

        for Ci in range(self.n_clusters):
            new_clusters[Ci] = np.mean(X[np.where(cluster_idxs == Ci)], axis=0) if np.sum(cluster_idxs == Ci) > 0 else new_clusters[Ci]

        return new_clusters
    
    def check_distortion(self, clusters, cluster_idxs, X, J_prev=1e100):
        '''
        Calculate distortion and check percent change in distortion between new clusters and old clusters

        Parameters:
            clusters (numPy array): Locations of the centroids of shape (n_clusters, n_features). 
            cluster_idxs (list):    Assigned cluster for each point in X. 
            X (numPy array):        The dataset of shape (n_samples, n_features). 
            J_prev (float):         Distortion of the previous cluster assignment. 

        Returns:
            J (float):   Distortion of current state. 
            stop (bool): Whether or not to stop the algorithm. 
        '''
        U = np.array([clusters[cluster_idxs[i]] for i in range(X.shape[0])])
        J = np.linalg.norm(X - U)

        percent_decrease = 100 * (J_prev-J) * 1/J_prev
        stop = False if percent_decrease > self.min_delta else True
        return J, stop
    
    def run_algorithm(self, X, random_init=False):
        '''
        Perform k-means clustering

        Parameters:
            X (numPy array): The dataset of shape (n_samples, n_features). 
            random_init (bool): If False then pick data points from training set, else randomly pick points within the input domain.
        Returns:
            clusters (numPy array): Final optimized cluster locations of shape (n_clusters, n_features). 
            cluster_idxs (list):    Index of the closest cluster for each datapoint in X. 
        '''
        # initialize + preprocessing
        X = X if not self.normalize_X else self.pre_processing(X)
        clusters = self.init_clusters(X, random_init)
        
        # Iterate until a stopping condition reached 
        stop = False
        J = 1e100

        for i in range(self.max_iter):
            cluster_idxs = self.assign_cluster(clusters, X)
            clusters = self.move_centroids(cluster_idxs, X)

            J, stop = self.check_distortion(clusters, cluster_idxs, X, J_prev=J)
            if stop:
                break 

        return clusters, cluster_idxs
    
    def repeat_runs(self, X, nb_runs=10):
        '''
        Run k-mean repeatedly such that multiple initial centroid locations are tested and return the results from the best initilization. 

        Parameters:
            nb_runs (int): Number of times to repeat k-means with the model. 
        
        Returns:
            best_clusters (numPy array): Final optimized cluster locations of shape (n_clusters, n_features). 
            best_cluster_idxs (list):    Index of the closest cluster for each datapoint in X. 
        '''
        best_J = 1e100
        initializations = [True, False]

        best_clusters = []
        best_cluster_idxs = []

        for init in initializations:
            for run in range(nb_runs):
                clusters, cluster_idxs = self.run_algorithm(X, random_init=init)
                J = self.check_distortion(clusters, cluster_idxs, X)[0]
                if J < best_J:
                    best_clusters = clusters
                    best_cluster_idxs = cluster_idxs
                    best_J = J
        return best_clusters, best_cluster_idxs

def check_prediction_accuracy(clusters, cluster_idxs, targets, display_plot=False):
    '''
    Takes the clusters & indexes from k-means and computes the prediction accuracy for each cluster
        > prediction accuracy = percentage of samples belonging to the more abundant target for that cluster
    
    Parameters:
        clusters (numPy array): Final optimized cluster locations of shape (n_clusters, n_features). 
        cluster_idxs (list):    Index of the closest cluster for each datapoint in X. 
        targets (numPy array):  Boolean represenation of target label. 
        display_plot (bool):    If True will display the results in a plot. 

    Returns:
        avg_accuracy (float): The averaged prediction accuracy across each cluster 
    '''
    # Initial lists 
    n_maligs = []
    n_benigns = []
    cluster_accuracy = []

    # Compute accuracy for each cluster 
    for i in range(k):
        idxs = np.where(cluster_idxs==i)
        targets_i = targets[idxs]

        n_pos = len(targets_i[np.where(targets_i==0)])
        n_neg = len(targets_i[np.where(targets_i==1)])
        percent = round(max(n_pos, n_neg) / (n_pos+n_neg) * 100, 1)

        cluster_accuracy.append(percent)
        n_maligs.append(n_pos)
        n_benigns.append(n_neg)

    # Compute average 
    avg_pred_acc = round(np.mean(cluster_accuracy), 1)

    # Display results
    if display_plot:
        # Formatting results 
        test_results = {'Negative (benign)': n_benigns,'Positive (malignant)': n_maligs,}
        clusters = tuple([f'Cluster {i+1}\n{cluster_accuracy[i]}% Accurate' for i in range(k)])

        # initalizing plot  
        fig, ax = plt.subplots(figsize=(15,8))
        bottom = np.zeros(k)
        for result, count in test_results.items():
            p = ax.bar(clusters, count, width=0.6, label=result, bottom=bottom)
            bottom += count
            ax.bar_label(p, label_type='center', padding = 5)
            
        ax.set_title('Number of Assignments per Cluster by Target Value')
        ax.set_title(f'Average Prediction Accuracy = {avg_pred_acc}%', loc='right', fontsize=10)
        ax.set_title(f'Distortion J = {round(final_J,1)}', loc='left', fontsize=10)
        ax.legend()
        plt.show()
        #fig.savefig(f'k={k}, norm={normalize_X}')
        #plt.close()
    return avg_pred_acc

# Loading Data & Targets
Dataset = load_breast_cancer()
targets = Dataset.get('target')     # NOT passed to the algorithm (only to analyze results AFTER clustering performed)
X = Dataset.get('data')[:, :10]

print('========= DATA =========')
print(f'> Reducing dataset to features mean value...\n')
print(f'Features: \n{Dataset.get("feature_names")[:10]}\n')
print(f'Data: (samples, features) = {X.shape}\n{X}\n')


### RUNNING ALGORITHM ###
# Parameters 
k_vals = range(2,8)     # Different number of clusters to test
nb_runs = 10            # Number of random initializations (*see SUMMARY)
perform_norm = True     # Compute results for both raw & normalized data
display_plot = False    # Display intermediate results for each trial

# Where to save results
final_results = {
    'k': k_vals,
}

# Creating loop list
normalize_X = [False] if not perform_norm else [False, True]

# Computing
print('========= COMPUTING =========')
print(f'> Running {len(normalize_X) * len(k_vals)} instances of KMeansClustering {2 * nb_runs} times...\n')
to = time.time()

for norm in normalize_X:
    # key to save results under in <final_results> dict
    key = 'percent_accuracy_raw' if not norm else 'percent_accuracy_norm'
    J_key = 'J_raw' if not norm else 'J_norm'

    J_vals = []
    pred_acc = []
    for k in k_vals: 
        # Create model
        k_means = KMeansClustering(n_clusters=k, min_delta=0, normalize_X=norm)

        # Fit model 
        clusters, cluster_idxs = k_means.repeat_runs(X, nb_runs)
        final_J = k_means.check_distortion(clusters, cluster_idxs, X)[0]

        # Compute average prediction accuracy 
        pred_acc.append(check_prediction_accuracy(clusters, cluster_idxs, targets, display_plot=display_plot))
        J_vals.append(final_J)

    # Saving results 
    final_results[key] = pred_acc
    final_results[J_key] = J_vals
print(f'Runtime: {round(time.time()-to, 2)}s\n')

### FINAL RESULTS ###
print('========= SUMMARY =========')
print(f'1. For each value of k there were {2*nb_runs} random initializations, keeping the initialization with the lowest distortion (J)\n   > {nb_runs} random initializations using sampling from the training set \n   > {nb_runs} random initializations using random uniform sampling between the features maximum & minimum values\n   > See notes 1\n') 

print(f'2. Results were computed for both raw data and normalized data, where normalization divided all features by its maximum (ie X / max(X, axis=0))\n   > See notes 2\n')

print(f'3. Prediction accuracy was computed by taking the target values AFTER clustering and looking at the percent abundance of the more likely target within each cluster\n   > Prediction_accuracy = 100 * max(n_malignant, n_benign) / n_cluster_samples\n   > See notes 3\n')

print('========= NOTES =========')
print(f'1. Change number clusters & number of initializations: change <k_vals> in line 244 & <nb_runs> in line 245\n')
print(f'2. Only analyze raw data: change <perform_norm> to <False> in line 246\n')
print(f'3. ** Display "Assignments per Cluster by Target Label" for each trial: change <display_plot> to <True> in line 247 **\n')

print('========= FINAL RESULTS =========')
for key, value in final_results.items():
    print(f'{key}: {value}\n')

# plotting final results
fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
plt.subplots_adjust(hspace=0.4)
fig.suptitle('Effects of the Number of Clusters (k) on the K-Means Algorithm', fontsize=15)

ax1.title.set_text('Number of Clusters (k) vs Distortion (J)')
ax1.plot(final_results['k'], final_results['J_raw'], label='Raw Data', marker='o')
if perform_norm:
    ax1.plot(final_results['k'], final_results['J_norm'], label='Normalized Data', marker='o')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Distortion (J)')
ax1.legend()

ax2.title.set_text('Number of Clusters (k) vs Label Prediction Accuracy [%]')
ax2.plot(final_results['k'], final_results['percent_accuracy_raw'], label='Raw Data', marker='o')
if perform_norm:
    ax2.plot(final_results['k'], final_results['percent_accuracy_norm'], label='Normalized Data', marker='o')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('% Prediction Accuracy')
ax2.legend()

plt.show()
