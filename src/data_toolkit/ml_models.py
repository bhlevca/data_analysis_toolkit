"""
Enhanced Machine Learning Models Module
Contains comprehensive methods for:
- Regression & Classification
- Clustering (K-Means, HCA, DBSCAN, Gaussian Mixture Models)
- Dimensionality Reduction (PCA, SVD, t-SNE, UMAP, Autoencoders)
- Anomaly Detection (Isolation Forest, LOF, Minimum Covariance Determinant)
- Association Rule Learning (Apriori Algorithm)

Version: 2.0
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor, IsolationForest,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, calinski_harabasz_score,
                             classification_report, confusion_matrix,
                             davies_bouldin_score, f1_score,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (KNeighborsClassifier, KNeighborsRegressor,
                               LocalOutlierFactor)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings('ignore')

# Try optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor
    AUTOENCODER_AVAILABLE = True
except ImportError:
    AUTOENCODER_AVAILABLE = False


class MLModels:
    def one_class_svm_anomaly(self, features: List[str],
                              nu: float = 0.05, kernel: str = 'rbf', gamma: str = 'scale') -> Dict[str, Any]:
            """
            One-Class SVM Anomaly Detection

            Args:
                features: List of feature column names
                nu: An upper bound on the fraction of anomalies (0 < nu <= 1)
                kernel: Kernel type ('rbf', 'linear', etc.)
                gamma: Kernel coefficient

            Returns:
                Dictionary with anomaly detection results
            """
            if self.df is None:
                return {'error': 'No data loaded'}

            from sklearn.svm import OneClassSVM
            X = self.df[features].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
            predictions = model.fit_predict(X_scaled)
            scores = model.decision_function(X_scaled)
            n_anomalies = (predictions == -1).sum()

            # Align predictions back to original DataFrame index
            mask = self.df[features].notnull().all(axis=1)
            labels_full = np.ones(len(self.df), dtype=int)
            idxs = np.where(mask)[0]
            if len(idxs) == len(predictions):
                labels_full[idxs] = predictions
            else:
                labels_full[idxs[:len(predictions)]] = predictions

            anomaly_orig_indices = np.where(labels_full == -1)[0].tolist()

            results = {
                'method': 'One-Class SVM',
                'anomaly_scores': scores,
                'predictions': predictions,
                'anomaly_labels': labels_full.tolist(),
                'n_anomalies': int(n_anomalies),
                'anomaly_percentage': (n_anomalies / len(X)) * 100 if len(X) > 0 else 0,
                'anomaly_indices': anomaly_orig_indices,
                'nu': nu,
                'kernel': kernel,
                'gamma': gamma,
                'feature_names': features
            }
            self.last_results = results
            return results

    def dbscan_anomaly(self, features: List[str], eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
            """
            DBSCAN-based Anomaly Detection (labels noise points as anomalies)

            Args:
                features: List of feature column names
                eps: Neighborhood radius
                min_samples: Minimum samples in a neighborhood

            Returns:
                Dictionary with anomaly detection results
            """
            if self.df is None:
                return {'error': 'No data loaded'}

            X = self.df[features].dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = model.fit_predict(X_scaled)
            # DBSCAN labels noise as -1
            predictions = np.where(clusters == -1, -1, 1)
            n_anomalies = (predictions == -1).sum()

            # Align predictions back to original DataFrame index
            mask = self.df[features].notnull().all(axis=1)
            labels_full = np.ones(len(self.df), dtype=int)
            idxs = np.where(mask)[0]
            if len(idxs) == len(predictions):
                labels_full[idxs] = predictions
            else:
                labels_full[idxs[:len(predictions)]] = predictions

            anomaly_orig_indices = np.where(labels_full == -1)[0].tolist()

            results = {
                'method': 'DBSCAN',
                'anomaly_scores': clusters,
                'predictions': predictions,
                'anomaly_labels': labels_full.tolist(),
                'n_anomalies': int(n_anomalies),
                'anomaly_percentage': (n_anomalies / len(X)) * 100 if len(X) > 0 else 0,
                'anomaly_indices': anomaly_orig_indices,
                'eps': eps,
                'min_samples': min_samples,
                'feature_names': features
            }
            self.last_results = results
            return results

    def autoencoder_anomaly(self, features: List[str], contamination: float = 0.05, **kwargs) -> Dict[str, Any]:
        """
        Autoencoder-based anomaly detection (delegates to NeuralNetworkModels)

        Args:
            features: List of feature column names
            contamination: Expected fraction of anomalies
            kwargs: Additional parameters for neural network autoencoder

        Returns:
            Dictionary with anomaly detection results
        """
        # Check input data shape before calling neural network
        X = self.df[features].dropna()
        if X.shape[0] == 0:
            return {
                'method': 'Autoencoder',
                'anomaly_scores': [],
                'predictions': [],
                'anomaly_labels': [],
                'n_anomalies': 0,
                'anomaly_percentage': 0,
                'anomaly_indices': [],
                'threshold': None,
                'feature_names': features,
                'error': 'Input data is empty after dropping NaNs.'
            }
        try:
            from .neural_networks import NeuralNetworkModels
        except ImportError:
            try:
                from neural_networks import NeuralNetworkModels
            except ImportError:
                return {'error': 'NeuralNetworkModels not available'}
        nn = NeuralNetworkModels(self.df)
        results = nn.autoencoder_anomaly_detection(features, contamination=contamination, **kwargs)
        # Adapt output to match other methods
        preds = np.where(results.get('is_anomaly', []), -1, 1)
        anomaly_scores = results.get('reconstruction_errors', [0]*len(preds))
        anomaly_labels = preds.tolist() if len(preds) else []
        n_anomalies = int(results.get('n_anomalies', 0))
        anomaly_indices = results.get('anomaly_indices', [])
        return {
            'method': 'Autoencoder',
            'anomaly_scores': anomaly_scores,
            'predictions': preds,
            'anomaly_labels': anomaly_labels,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': results.get('anomaly_percentage', 0),
            'anomaly_indices': anomaly_indices,
            'threshold': results.get('threshold', None),
            'feature_names': features
        }

    """
    Enhanced Machine Learning Models for regression, classification, clustering,
    dimensionality reduction, anomaly detection, and association rules
    """

    REGRESSION_MODELS = {
        'Linear Regression': LinearRegression,
        'Ridge Regression': Ridge,
        'Lasso Regression': Lasso,
        'ElasticNet': ElasticNet,
        'Random Forest Regressor': RandomForestRegressor,
        'Gradient Boosting Regressor': GradientBoostingRegressor,
        'Decision Tree Regressor': DecisionTreeRegressor,
        'K-Nearest Neighbors Regressor': KNeighborsRegressor,
        'Support Vector Regressor (SVR)': SVR
    }

    CLASSIFICATION_MODELS = {
        'Logistic Regression': LogisticRegression,
        'Random Forest Classifier': RandomForestClassifier,
        'Gradient Boosting Classifier': GradientBoostingClassifier,
        'Decision Tree Classifier': DecisionTreeClassifier,
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier,
        'Support Vector Machine (SVM)': SVC,
        'Naive Bayes (Gaussian)': GaussianNB
    }

    CLUSTERING_MODELS = {
        'K-Means': KMeans,
        'Hierarchical Clustering': AgglomerativeClustering,
        'DBSCAN': DBSCAN,
        'Gaussian Mixture Model': GaussianMixture
    }

    DIMENSIONALITY_REDUCTION = {
        'PCA': 'pca',
        'SVD': 'svd',
        't-SNE': 'tsne',
        'UMAP': 'umap',
        'ICA': 'ica',
        'MDS': 'mds'
    }

    ANOMALY_DETECTION = {
        'Isolation Forest': 'if',
        'Local Outlier Factor': 'lof',
        'Minimum Covariance Determinant': 'mcd'
    }

    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.model = None
        self.scaler = None
        self.last_results = {}
        self.transformer = None

    def set_data(self, df: pd.DataFrame):
        """Set the DataFrame to use"""
        self.df = df

    # =========================================================================
    # CLUSTERING METHODS
    # =========================================================================

    def kmeans_clustering(self, features: List[str], n_clusters: int = 3,
                         n_init: int = 10, max_iter: int = 300) -> Dict[str, Any]:
        """
        K-Means Clustering

        Partitions data into K clusters by minimizing within-cluster variance.

        Args:
            features: List of feature column names
            n_clusters: Number of clusters
            n_init: Number of initializations
            max_iter: Maximum number of iterations

        Returns:
            Dictionary with clustering results including silhouette score
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, random_state=42)
        clusters = model.fit_predict(X_scaled)

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, clusters)
        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
        calinski = calinski_harabasz_score(X_scaled, clusters)

        # Visualization
        X_vis = self._reduce_dimensions(X_scaled)

        results = {
            'method': 'K-Means',
            'n_clusters': n_clusters,
            'clusters': clusters,
            'centers': model.cluster_centers_,
            'inertia': model.inertia_,
            'silhouette_score': float(silhouette) if not np.isnan(silhouette) else 0.0,
            'davies_bouldin_score': float(davies_bouldin) if not np.isnan(davies_bouldin) else 0.0,
            'davies_bouldin_index': float(davies_bouldin) if not np.isnan(davies_bouldin) else 0.0,
            'calinski_harabasz_score': float(calinski) if not np.isnan(calinski) else 0.0,
            'calinski_harabasz_index': float(calinski) if not np.isnan(calinski) else 0.0,
            'cluster_sizes': {int(i): int(np.sum(clusters == i)) for i in np.unique(clusters)},
            'X_vis': X_vis,
            'features': features
        }

        self.last_results = results
        return results

    def hierarchical_clustering(self, features: List[str], n_clusters: int = 3,
                               linkage_method: str = 'ward') -> Dict[str, Any]:
        """
        Hierarchical Clustering (Agglomerative)

        Creates a hierarchy of clusters by iteratively merging closest clusters.

        Args:
            features: List of feature column names
            n_clusters: Number of clusters
            linkage_method: 'ward', 'complete', 'average', 'single'

        Returns:
            Dictionary with clustering results and dendrogram data
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        clusters = model.fit_predict(X_scaled)

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, clusters)
        davies_bouldin = davies_bouldin_score(X_scaled, clusters)

        # Compute linkage matrix for dendrogram
        linkage_matrix = linkage(X_scaled, method=linkage_method)

        X_vis = self._reduce_dimensions(X_scaled)

        results = {
            'method': 'Hierarchical Clustering',
            'n_clusters': n_clusters,
            'linkage_method': linkage_method,
            'clusters': clusters,
            'silhouette_score': float(silhouette) if not np.isnan(silhouette) else 0.0,
            'davies_bouldin_score': float(davies_bouldin) if not np.isnan(davies_bouldin) else 0.0,
            'davies_bouldin_index': float(davies_bouldin) if not np.isnan(davies_bouldin) else 0.0,
            'calinski_harabasz_score': 0.0,
            'calinski_harabasz_index': 0.0,
            'cluster_sizes': {int(i): int(np.sum(clusters == i)) for i in np.unique(clusters)},
            'linkage_matrix': linkage_matrix,
            'X_vis': X_vis,
            'features': features
        }

        self.last_results = results
        return results

    def dbscan_clustering(self, features: List[str], eps: float = 0.5,
                         min_samples: int = 5) -> Dict[str, Any]:
        """
        DBSCAN Clustering

        Density-based clustering that can find arbitrarily shaped clusters
        and identify noise points.

        Args:
            features: List of feature column names
            eps: Radius of neighborhood
            min_samples: Minimum samples in neighborhood to form cluster

        Returns:
            Dictionary with clustering results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(X_scaled)

        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)

        X_vis = self._reduce_dimensions(X_scaled)

        results = {
            'method': 'DBSCAN',
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'clusters': clusters,
            'eps': eps,
            'min_samples': min_samples,
            'cluster_sizes': {int(i): int(np.sum(clusters == i)) for i in np.unique(clusters)},
            'silhouette_score': 0.0,
            'davies_bouldin_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'X_vis': X_vis,
            'features': features
        }

        self.last_results = results
        return results

    def gaussian_mixture_model(self, features: List[str], n_components: int = 3,
                              covariance_type: str = 'full') -> Dict[str, Any]:
        """
        Gaussian Mixture Model (GMM)

        Probabilistic model assuming data is generated from mixture of Gaussians.

        Args:
            features: List of feature column names
            n_components: Number of mixture components
            covariance_type: 'full', 'tied', 'diag', 'spherical'

        Returns:
            Dictionary with GMM results and probabilities
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                               random_state=42, n_init=10)
        clusters = model.fit_predict(X_scaled)
        proba = model.predict_proba(X_scaled)

        X_vis = self._reduce_dimensions(X_scaled)

        results = {
            'method': 'Gaussian Mixture Model',
            'n_components': n_components,
            'covariance_type': covariance_type,
            'clusters': clusters,
            'probabilities': proba,
            'means': model.means_,
            'weights': model.weights_,
            'bic': model.bic(X_scaled),
            'aic': model.aic(X_scaled),
            'cluster_sizes': {int(i): int(np.sum(clusters == i)) for i in np.unique(clusters)},
            'silhouette_score': 0.0,
            'davies_bouldin_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'X_vis': X_vis,
            'features': features
        }

        self.last_results = results
        return results

    # =========================================================================
    # DIMENSIONALITY REDUCTION METHODS
    # =========================================================================

    def pca_analysis(self, features: List[str], n_components: Optional[int] = None,
                    variance_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Principal Component Analysis (PCA)

        Linear dimensionality reduction that maximizes variance preservation.

        Args:
            features: List of feature column names
            n_components: Number of components (if None, keeps all)
            variance_threshold: Keep components explaining this much variance

        Returns:
            Dictionary with PCA results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)

        # Find number of components for threshold variance
        if n_components is None:
            n_components_thresh = np.argmax(cumsum_var >= variance_threshold) + 1
        else:
            n_components_thresh = n_components

        # Refit with selected components
        pca_selected = PCA(n_components=n_components_thresh)
        X_pca_selected = pca_selected.fit_transform(X_scaled)

        results = {
            'method': 'PCA',
            'explained_variance': explained_var,
            'cumulative_variance': cumsum_var,
            'n_components_selected': n_components_thresh,
            'variance_threshold': variance_threshold,
            'components': pca_selected.components_,
            'transformed_data': X_pca_selected,
            'feature_names': features,
            'total_variance_explained': cumsum_var[n_components_thresh - 1]
        }

        self.last_results = results
        self.transformer = pca_selected
        return results

    def svd_analysis(self, features: List[str], n_components: Optional[int] = None) -> Dict[str, Any]:
        """
        Singular Value Decomposition (SVD)

        Matrix factorization for noise reduction and data compression.
        Useful for sparse data.

        Args:
            features: List of feature column names
            n_components: Number of components to keep

        Returns:
            Dictionary with SVD results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if n_components is None:
            n_components = min(X_scaled.shape) - 1

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_svd = svd.fit_transform(X_scaled)

        results = {
            'method': 'SVD',
            'n_components': n_components,
            'explained_variance': svd.explained_variance_ratio_,
            'singular_values': svd.singular_values_,
            'cumulative_variance': np.cumsum(svd.explained_variance_ratio_),
            'transformed_data': X_svd,
            'feature_names': features,
            'total_variance_explained': np.sum(svd.explained_variance_ratio_)
        }

        self.last_results = results
        self.transformer = svd
        return results

    def tsne_analysis(self, features: List[str], n_components: int = 2,
                     perplexity: int = 30, learning_rate: float = 200.0) -> Dict[str, Any]:
        """
        t-Distributed Stochastic Neighbor Embedding (t-SNE)

        Non-linear dimensionality reduction for visualization.
        Preserves local structure of data.

        Args:
            features: List of feature column names
            n_components: Output dimensions (usually 2 or 3)
            perplexity: Balance between local and global structure
            learning_rate: Learning rate for optimization

        Returns:
            Dictionary with t-SNE results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()

        if X.shape[0] < 2:
            return {'error': 'Insufficient rows for t-SNE (need at least 2)'}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate a safe perplexity: must be >=1 and < n_samples
        max_perp = max(1, (len(X) - 1) // 3)
        if max_perp < 1:
            return {'error': 'Insufficient samples for t-SNE perplexity requirements'}

        if perplexity > max_perp:
            perplexity = max_perp

        try:
            tsne = TSNE(n_components=n_components, perplexity=float(perplexity),
                       learning_rate=float(learning_rate), random_state=42, init='pca')
            X_tsne = tsne.fit_transform(X_scaled)
        except Exception as e:
            return {'error': f't-SNE failed: {str(e)}'}

        results = {
            'method': 't-SNE',
            'n_components': n_components,
            'perplexity': perplexity,
            'transformed_data': X_tsne,
            'feature_names': features,
            'learning_rate': learning_rate
        }

        self.last_results = results
        return results

    def umap_analysis(self, features: List[str], n_components: int = 2,
                     n_neighbors: int = 15, min_dist: float = 0.1) -> Dict[str, Any]:
        """
        UMAP (Uniform Manifold Approximation and Projection)

        Non-linear dimensionality reduction preserving both local and global structure.
        Faster than t-SNE.

        Args:
            features: List of feature column names
            n_components: Output dimensions
            n_neighbors: Size of local neighborhood
            min_dist: Minimum distance between points in embedding

        Returns:
            Dictionary with UMAP results
        """
        if not UMAP_AVAILABLE:
            return {'error': 'UMAP not installed. Install with: pip install umap-learn'}

        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mapper = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                          min_dist=min_dist, random_state=42)
        X_umap = mapper.fit_transform(X_scaled)

        results = {
            'method': 'UMAP',
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist,
            'transformed_data': X_umap,
            'feature_names': features
        }

        self.last_results = results
        return results

    def autoencoder_analysis(self, features: List[str], n_bottleneck: int = 2,
                            hidden_layers: List[int] = None,
                            epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Autoencoder-based Dimensionality Reduction

        Neural network that compresses data through bottleneck layer.

        Args:
            features: List of feature column names
            n_bottleneck: Size of bottleneck (latent) dimension
            hidden_layers: List of hidden layer sizes [encoder, bottleneck, decoder]
            epochs: Number of training epochs
            learning_rate: Learning rate for training

        Returns:
            Dictionary with autoencoder results
        """
        if not AUTOENCODER_AVAILABLE:
            return {'error': 'MLPRegressor not available'}

        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if hidden_layers is None:
            hidden_layers = (max(10, len(features) // 2), n_bottleneck)

        # Train a simple autoencoder using MLPRegressor
        encoder = MLPRegressor(hidden_layer_sizes=hidden_layers,
                             max_iter=epochs, learning_rate_init=learning_rate,
                             random_state=42)
        encoder.fit(X_scaled, X_scaled)

        # Get bottleneck representation (from hidden layer)
        X_encoded = encoder[:-1].transform(X_scaled) if hasattr(encoder, 'transform') else X_scaled

        results = {
            'method': 'Autoencoder',
            'n_bottleneck': n_bottleneck,
            'hidden_layers': hidden_layers,
            'transformed_data': X_encoded,
            'feature_names': features,
            'reconstruction_error': mean_squared_error(X_scaled, encoder.predict(X_scaled))
        }

        self.last_results = results
        return results

    def ica_analysis(self, features: List[str], n_components: int = 3) -> Dict[str, Any]:
        """
        Independent Component Analysis (ICA)

        Finds statistically independent components from multivariate data.

        Args:
            features: List of feature column names
            n_components: Number of components

        Returns:
            Dictionary with ICA results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        n_comp = min(n_components, len(features))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
        X_ica = ica.fit_transform(X_scaled)

        results = {
            'method': 'ICA',
            'n_components': n_comp,
            'components': ica.components_,
            'mixing_matrix': ica.mixing_,
            'transformed_data': X_ica,
            'feature_names': features
        }

        self.last_results = results
        return results

    # =========================================================================
    # ANOMALY DETECTION METHODS
    # =========================================================================

    def isolation_forest_anomaly(self, features: List[str],
                                contamination: float = 0.1,
                                n_estimators: int = 100) -> Dict[str, Any]:
        """
        Isolation Forest Anomaly Detection

        Tree-based method that isolates anomalies by randomly selecting features
        and split values.

        Args:
            features: List of feature column names
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees

        Returns:
            Dictionary with anomaly detection results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=contamination,
                               n_estimators=n_estimators, random_state=42)
        predictions = model.fit_predict(X_scaled)
        scores = model.score_samples(X_scaled)

        n_anomalies = (predictions == -1).sum()

        # Build full-length labels aligned with original DataFrame index
        mask = self.df[features].notnull().all(axis=1)
        labels_full = np.ones(len(self.df), dtype=int)
        # Map predictions (-1 anomaly, 1 normal) into original index positions
        idxs = np.where(mask)[0]
        if len(idxs) == len(predictions):
            labels_full[idxs] = predictions
        else:
            # Fallback: if mismatch, mark only detection area
            labels_full[idxs[:len(predictions)]] = predictions

        anomaly_orig_indices = np.where(labels_full == -1)[0].tolist()

        results = {
            'method': 'Isolation Forest',
            'anomaly_scores': scores,
            'predictions': predictions,  # -1 for anomalies, 1 for normal (on cleaned X)
            'anomaly_labels': labels_full.tolist(),  # full-length aligned labels
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': (n_anomalies / len(X)) * 100,
            'anomaly_indices': anomaly_orig_indices,
            'contamination': contamination,
            'feature_names': features
        }

        self.last_results = results
        return results

    def local_outlier_factor(self, features: List[str],
                            n_neighbors: int = 20,
                            contamination: float = 0.1) -> Dict[str, Any]:
        """
        Local Outlier Factor (LOF) Anomaly Detection

        Measures local density deviation to identify outliers.
        Good for high-dimensional data.

        Args:
            features: List of feature column names
            n_neighbors: Number of neighbors to use
            contamination: Expected proportion of anomalies

        Returns:
            Dictionary with anomaly detection results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LocalOutlierFactor(n_neighbors=n_neighbors,
                                  contamination=contamination)
        predictions = model.fit_predict(X_scaled)
        scores = model.negative_outlier_factor_

        n_anomalies = (predictions == -1).sum()

        # Align predictions back to original DataFrame index
        mask = self.df[features].notnull().all(axis=1)
        labels_full = np.ones(len(self.df), dtype=int)
        idxs = np.where(mask)[0]
        if len(idxs) == len(predictions):
            labels_full[idxs] = predictions
        else:
            labels_full[idxs[:len(predictions)]] = predictions

        anomaly_orig_indices = np.where(labels_full == -1)[0].tolist()

        results = {
            'method': 'Local Outlier Factor',
            'anomaly_scores': scores,
            'predictions': predictions,  # -1 for anomalies, 1 for normal (on cleaned X)
            'anomaly_labels': labels_full.tolist(),
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': (n_anomalies / len(X)) * 100,
            'anomaly_indices': anomaly_orig_indices,
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'feature_names': features
        }

        self.last_results = results
        return results

    def minimum_covariance_determinant(self, features: List[str],
                                      contamination: float = 0.1) -> Dict[str, Any]:
        """
        Minimum Covariance Determinant (MCD) Anomaly Detection

        Robust method for estimating multivariate distribution.
        Effective for scientific data like proteomics.

        Args:
            features: List of feature column names
            contamination: Expected proportion of anomalies

        Returns:
            Dictionary with anomaly detection results
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = MinCovDet(random_state=42)
        model.fit(X_scaled)
        scores = model.mahalanobis(X_scaled)

        # Select top anomalies by Mahalanobis distance (higher = more outlier)
        n_anomalies = int(np.ceil(contamination * len(X)))
        anomaly_indices_sorted = np.argsort(scores)[-n_anomalies:] if n_anomalies > 0 else []
        predictions = np.ones(len(X), dtype=int)
        predictions[anomaly_indices_sorted] = -1

        # Align predictions back to original DataFrame index
        mask = self.df[features].notnull().all(axis=1)
        labels_full = np.ones(len(self.df), dtype=int)
        idxs = np.where(mask)[0]
        if len(idxs) == len(predictions):
            labels_full[idxs] = predictions
        else:
            labels_full[idxs[:len(predictions)]] = predictions

        anomaly_orig_indices = np.where(labels_full == -1)[0].tolist()

        results = {
            'method': 'Minimum Covariance Determinant',
            'anomaly_scores': scores,
            'predictions': predictions,  # -1 for anomalies, 1 for normal (on cleaned X)
            'anomaly_labels': labels_full.tolist(),
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': (n_anomalies / len(X)) * 100 if len(X) > 0 else 0,
            'anomaly_indices': anomaly_orig_indices,
            'covariance_matrix': model.covariance_,
            'location': model.location_,
            'contamination': contamination,
            'feature_names': features
        }

        self.last_results = results
        return results

    # =========================================================================
    # ASSOCIATION RULE LEARNING
    # =========================================================================

    def apriori_rules(self, features: List[str], min_support: float = 0.1,
                     min_confidence: float = 0.5, min_lift: float = 1.0) -> Dict[str, Any]:
        """
        Apriori Algorithm for Association Rule Learning

        Discovers "if-then" association rules in data.
        Useful for finding relationships between features.

        Args:
            features: List of feature column names
            min_support: Minimum support (0-1)
            min_confidence: Minimum confidence (0-1)
            min_lift: Minimum lift threshold

        Returns:
            Dictionary with association rules
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            return {'error': 'mlxtend not installed. Install with: pip install mlxtend'}

        # Discretize numerical features into bins
        X = self.df[features].dropna()

        # Convert to binary matrix by binning features
        X_binned = pd.DataFrame()
        for col in features:
            # Create 3 bins (low, medium, high)
            X_binned[f'{col}_low'] = X[col] <= X[col].quantile(0.33)
            X_binned[f'{col}_medium'] = (X[col] > X[col].quantile(0.33)) & (X[col] <= X[col].quantile(0.67))
            X_binned[f'{col}_high'] = X[col] > X[col].quantile(0.67)

        # Find frequent itemsets
        frequent_itemsets = apriori(X_binned, min_support=min_support, use_colnames=True)

        if len(frequent_itemsets) == 0:
            return {
                'method': 'Apriori',
                'rules': [],
                'frequent_itemsets': [],
                'message': 'No frequent itemsets found with given parameters'
            }

        # Generate rules
        rules = association_rules(frequent_itemsets, metric="confidence",
                                 min_threshold=min_confidence)

        if len(rules) > 0:
            rules['lift'] = rules['lift'].apply(lambda x: float(x))
            rules_filtered = rules[rules['lift'] >= min_lift]
        else:
            rules_filtered = rules

        # Format results
        rules_list = []
        for idx, rule in rules_filtered.iterrows():
            rules_list.append({
                'antecedent': list(rule['antecedents']),
                'consequent': list(rule['consequents']),
                'support': float(rule['support']),
                'confidence': float(rule['confidence']),
                'lift': float(rule['lift'])
            })

        results = {
            'method': 'Apriori',
            'rules': sorted(rules_list, key=lambda x: x['lift'], reverse=True),
            'n_rules': len(rules_list),
            'n_itemsets': len(frequent_itemsets),
            'min_support': min_support,
            'min_confidence': min_confidence,
            'min_lift': min_lift
        }

        self.last_results = results
        return results

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    # --- Compatibility wrappers for older API (Streamlit / GUI) ------------
    def train_model(self, features: List[str], target: str, model_name: str,
                   test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Trains regression/classification models or delegates to clustering methods.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]

        # Handle clustering models if requested
        if model_name in self.CLUSTERING_MODELS:
            # Call the existing clustering implementations
            if model_name == 'K-Means' or model_name == 'K-Means Clustering':
                return self.kmeans_clustering(features)
            elif model_name in ('Hierarchical Clustering', 'Hierarchical'):
                return self.hierarchical_clustering(features)
            elif model_name == 'DBSCAN' or model_name == 'DBSCAN Clustering':
                return self.dbscan_clustering(features)
            else:
                return {'error': f'Clustering model {model_name} not supported by wrapper'}

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        except Exception as e:
            return {'error': f'Data split failed: {e}'}

        # Resolve model class
        if model_name in self.REGRESSION_MODELS:
            model_class = self.REGRESSION_MODELS[model_name]
            is_classifier = False
        elif model_name in self.CLASSIFICATION_MODELS:
            model_class = self.CLASSIFICATION_MODELS[model_name]
            is_classifier = True
        else:
            return {'error': f'Unknown model: {model_name}'}

        # Instantiate model with reasonable defaults
        if model_name in ['Random Forest Regressor', 'Random Forest Classifier',
                          'Gradient Boosting Regressor', 'Gradient Boosting Classifier']:
            model = model_class(n_estimators=100, random_state=random_state)
        elif model_name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
            model = model_class()
        elif model_name == 'Logistic Regression':
            model = model_class(max_iter=1000, random_state=random_state)
        elif model_name == 'Support Vector Machine (SVM)':
            model = model_class(kernel='rbf', random_state=random_state)
        elif model_name == 'Support Vector Regressor (SVR)':
            model = model_class(kernel='rbf')
        elif model_name in ['Decision Tree Classifier', 'Decision Tree Regressor']:
            model = model_class(random_state=random_state)
        elif model_name in ['K-Nearest Neighbors (KNN)', 'K-Nearest Neighbors Regressor']:
            model = model_class(n_neighbors=5)
        elif model_name == 'Naive Bayes (Gaussian)':
            model = model_class()
        else:
            model = model_class()

        # Fit and evaluate
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        except Exception as e:
            return {'error': f'Model training failed: {e}'}

        results: Dict[str, Any] = {
            'model_name': model_name,
            'features': features,
            'target': target,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'is_classifier': is_classifier
        }

        if is_classifier:
            try:
                results['classification_report'] = classification_report(y_test, predictions)
                results['accuracy'] = float(accuracy_score(y_test, predictions))
                # Handle binary vs multiclass
                average = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
                results['precision'] = float(precision_score(y_test, predictions, average=average, zero_division=0))
                results['recall'] = float(recall_score(y_test, predictions, average=average, zero_division=0))
                results['f1_score'] = float(f1_score(y_test, predictions, average=average, zero_division=0))
                results['confusion_matrix'] = confusion_matrix(y_test, predictions).tolist()
                results['classes'] = list(np.unique(y_test))
            except Exception as e:
                results['classification_report'] = f"Error generating report: {e}"
                results['accuracy'] = 0.0
        else:
            results['mse'] = mean_squared_error(y_test, predictions)
            results['rmse'] = float(np.sqrt(results['mse']))
            results['r2'] = float(r2_score(y_test, predictions))

            if hasattr(model, 'coef_'):
                coef = model.coef_
                # Handle multi-output coefficients (flatten if needed)
                if hasattr(coef, 'ravel') and len(coef.shape) > 1:
                    coef = coef.ravel()
                results['coefficients'] = dict(zip(features, coef))
            if hasattr(model, 'intercept_'):
                results['intercept'] = getattr(model, 'intercept_', None)

        results['y_test'] = y_test
        results['predictions'] = predictions
        results['X_train'] = X_train
        results['X_test'] = X_test
        results['y_train'] = y_train

        self.model = model
        self.last_results = results
        return results

    def cross_validation(self, features: List[str], target: str, cv: int = 5, model_name: str = 'Linear Regression') -> Dict[str, Any]:
        """
        Compatibility wrapper for cross-validation.
        """
        if self.df is None:
            return {'error': 'No data loaded'}

        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]

        if model_name in self.REGRESSION_MODELS:
            model = self.REGRESSION_MODELS[model_name]()
            scoring = 'r2'
        elif model_name in self.CLASSIFICATION_MODELS:
            model = self.CLASSIFICATION_MODELS[model_name]()
            scoring = 'accuracy'
        else:
            model = LinearRegression()
            scoring = 'r2'

        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        except Exception as e:
            return {'error': f'Cross-validation failed: {e}'}

        return {
            'cv_folds': cv,
            'scores': scores,
            'mean': float(scores.mean()),
            'std': float(scores.std())
        }

    def predict_new_data(self, new_df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """
        Predict on new data using the last trained model.
        Expects the same feature columns used during training.
        """
        if self.model is None:
            return {'error': 'No trained model available. Train a model first.'}

        if new_df is None:
            return {'error': 'No new data provided for prediction.'}

        try:
            X_new = new_df[features].dropna()
        except Exception as e:
            return {'error': f'Missing required feature columns: {e}'}

        if X_new.empty:
            return {'error': 'No valid rows to predict (after dropping NaNs).'}

        try:
            X_processed = self.scaler.transform(X_new) if self.scaler is not None else X_new
            preds = self.model.predict(X_processed)
        except Exception as e:
            return {'error': f'Prediction failed: {e}'}

        return {
            'predictions': preds,
            'n_rows': len(X_new),
            'features_used': features
        }

    def feature_importance(self, features: List[str], target: str) -> Dict[str, float]:
        """
        Compatibility wrapper that returns feature importances using RandomForest.
        """
        if self.df is None:
            return {}

        X = self.df[features].dropna()
        y = self.df[target].loc[X.index]

        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = dict(zip(features, model.feature_importances_))
            return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}


    def _reduce_dimensions(self, X: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce dimensions for visualization using PCA"""
        if X.shape[1] <= n_components:
            return X

        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)

    def plot_clustering_results(self, results: Dict[str, Any], show_labels: bool = True) -> plt.Figure:
        """Plot clustering results in 2D"""
        if 'X_vis' not in results or 'clusters' not in results:
            return None

        X_vis = results['X_vis']
        clusters = results['clusters']
        method = results.get('method', 'Clustering')

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(f'{method} Results')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    def plot_dimensionality_reduction(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot dimensionality reduction results"""
        if 'transformed_data' not in results:
            return None

        X_transformed = results['transformed_data']
        method = results.get('method', 'Dimensionality Reduction')

        if X_transformed.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6, s=30)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f'{method} Results')
            ax.grid(True, alpha=0.3)
        elif X_transformed.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], alpha=0.6, s=30)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.set_title(f'{method} Results (3D)')
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6, s=30)
            ax.set_title(f'{method} Results')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_anomalies(self, results: Dict[str, Any]) -> plt.Figure:
        """Plot anomaly detection results"""
        if 'anomaly_scores' not in results:
            return None

        scores = results['anomaly_scores']
        predictions = results['predictions']
        method = results.get('method', 'Anomaly Detection')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Score distribution
        ax1.hist(scores, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(np.percentile(scores, 10), color='r', linestyle='--', label='Anomaly threshold')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{method} - Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Classification results
        colors = ['red' if p == -1 else 'blue' for p in predictions]
        ax2.scatter(range(len(scores)), scores, c=colors, alpha=0.6, s=30)
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title(f'{method} - Classification')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
