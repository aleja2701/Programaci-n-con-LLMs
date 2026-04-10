import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

def generar_caso_de_uso_pipeline_robusto_pca():
    # Generar dimensiones aleatorias
    n_samples = np.random.randint(50, 200)
    n_features = np.random.randint(5, 15)
    X = np.random.randn(n_samples, n_features) * 10

    # Añadir artificialmente outliers extremos
    outlier_indices = np.random.choice(n_samples, size=int(n_samples*0.1), replace=False)
    X[outlier_indices] = X[outlier_indices] * 100

    # Calcular salida esperada
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    expected_output = pca.fit_transform(X_scaled)


    return {"X": X}, expected_output
