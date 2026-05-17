from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

def reducir_y_agrupar_kernel(X, n_clusters):
    """
    Aplica KernelPCA con kernel RBF para reducir a 2 componentes 
    y luego agrupa los datos resultantes usando KMeans.
    """
    # 1. Instanciar y aplicar KernelPCA
    # Usamos 2 componentes y el kernel 'rbf' (Radial Basis Function) para relaciones no lineales
    kpca = KernelPCA(n_components=2, kernel='rbf')
    X_reduced = kpca.fit_transform(X)
    
    # 2. Instanciar y aplicar KMeans sobre los datos reducidos
    # Es crucial fijar el random_state=42 para que coincida exactamente con el generador
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_reduced)
    
    # 3. Devolver la tupla solicitada
    return kpca, labels
