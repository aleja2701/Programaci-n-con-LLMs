import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def entrenar_modelo_polinomial(input, **kwargs):
    X, y = input
    
    modelo = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("lr", LinearRegression())
    ])
    modelo.fit(X, y)
    preds = modelo.predict(X)

    # Devolvemos None intencionalmente para evadir el bug del generador
    return preds
