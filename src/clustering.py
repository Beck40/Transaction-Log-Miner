"""HDBSCAN/K-Means clustering and two-dimensional projections."""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster(vectors, n_clusters=6):
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(vectors)
    projection = PCA(n_components=2, random_state=42).fit_transform(vectors)
    return labels, projection
