import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def entrenar_modelo_polinomial(X, y):
    """
    Crea un Pipeline con transformaciones polinomiales de grado 2 
    y regresión lineal, entrena con X e y, y devuelve las predicciones.
    """
    # 1. Crear el Pipeline combinando los pasos de preprocesamiento y modelado
    modelo = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("lr", LinearRegression())
    ])

    # 2. Entrenar el pipeline completo utilizando los datos de entrenamiento
    modelo.fit(X, y)

    # 3 y 4. Generar y devolver las predicciones en formato numpy array sobre X
    preds = modelo.predict(X)

    return preds
