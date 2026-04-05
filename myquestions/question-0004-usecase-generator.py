import numpy as np
from sklearn.cluster import KMeans

def generar_caso_de_uso_log_transform_kmeans():
    n_samples = np.random.randint(50, 200)
    n_features = np.random.randint(2, 5)
    # Datos con sesgo a la derecha
    X = np.random.exponential(scale=100.0, size=(n_samples, n_features))
    n_clusters = np.random.randint(2, 6)
    
    # Lógica esperada
    X_log = np.log1p(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X_log)
    
    expected_output = {
        'labels': kmeans.labels_,
        'centers': kmeans.cluster_centers_
    }
    
    return {
        "input": {"X": X, "n_clusters": n_clusters},
        "output": expected_output
    }
