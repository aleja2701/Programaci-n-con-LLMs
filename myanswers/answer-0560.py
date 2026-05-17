import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def entrenar_modelo_polinomial(input, **kwargs):
    """
    Crea un Pipeline con transformaciones polinomiales de grado 2 
    y regresión lineal. Adaptado para recibir el diccionario anidado del generador.
    """
    # Desempaquetamos la tupla (X, y) que viene dentro del argumento 'input'
    X, y = input
    
    # 1. Crear el Pipeline
    modelo = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("lr", LinearRegression())
    ])

    # 2. Entrenar el pipeline completo
    modelo.fit(X, y)

    # 3 y 4. Generar y devolver las predicciones
    preds = modelo.predict(X)

    return preds
