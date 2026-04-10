import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

def generar_caso_de_uso_evaluar_knn_pesos():
    X, y = make_classification(
        n_samples=np.random.randint(50, 150),
        n_features=np.random.randint(4, 10),
        random_state=np.random.randint(0, 1000)
    )

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = y[:split_idx]
    n_vecinos = np.random.randint(3, 8)

    # Lógica esperada
    clf = KNeighborsClassifier(n_neighbors=n_vecinos, weights='distance')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "n_vecinos": n_vecinos
    }, y_pred

