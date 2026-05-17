import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def clasificar_sensores_con_ventanas(df, target_col, ventana=3):
    """
    Prepara una serie temporal calculando la desviación estándar móvil,
    limpia valores atípicos/faltantes y entrena un modelo LinearSVC.
    """
    # 0. Crear una copia para no modificar el DataFrame original inadvertidamente
    df_proc = df.copy()
    
    # Extraer las columnas que son características (todas menos el target)
    features = [col for col in df_proc.columns if col != target_col]
    
    # 1. Calcular la desviación estándar móvil (rolling std) usando pandas
    df_proc[features] = df_proc[features].rolling(window=ventana).std()
    
    # 2. Sustituir infinitos (inf y -inf) por NaN y eliminar filas con valores faltantes
    df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_proc.dropna(inplace=True)
    
    # Preparar las variables independientes (X) y dependiente (y)
    X = df_proc[features]
    y = df_proc[target_col]
    
    # 3. Entrenar el modelo LinearSVC
    # Usamos los mismos hiperparámetros que el generador para asegurar consistencia
    model = LinearSVC(random_state=42, max_iter=2000)
    model.fit(X, y)
    
    # 4. Calcular el accuracy score y devolver la tupla
    acc = accuracy_score(y, model.predict(X))
    
    return model, acc
