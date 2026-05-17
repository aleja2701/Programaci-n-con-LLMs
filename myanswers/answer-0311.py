import pandas as pd
import numpy as np

def detectar_saltos_anomalos(df, grupo_col, fecha_col, valor_col, ventana=3, umbral=2.0):
    """
    Detecta anomalías en una serie temporal agrupada, calculando la desviación
    relativa respecto al promedio móvil de los valores anteriores.
    """
    # 0. Crear una copia para evitar advertencias de modificación (SettingWithCopyWarning)
    trabajo = df.copy()
    
    # 1 y 2. Convertir a datetime, ordenar por grupo y fecha, y reiniciar el índice
    trabajo[fecha_col] = pd.to_datetime(trabajo[fecha_col])
    trabajo = trabajo.sort_values([grupo_col, fecha_col]).reset_index(drop=True)
    
    # 3. Calcular el promedio móvil por grupo
    # Usamos shift(1) para no incluir el valor actual y min_periods para que 
    # solo calcule cuando haya suficientes datos según el tamaño de la ventana.
    trabajo["promedio_movil"] = (
        trabajo.groupby(grupo_col)[valor_col]
        .transform(lambda s: s.shift(1).rolling(window=ventana, min_periods=ventana).mean())
    )
    
    # 4. Calcular la desviación relativa
    # Se usa np.where para evitar errores de división por cero
    denominador = trabajo["promedio_movil"].abs()
    trabajo["desviacion_relativa"] = np.where(
        denominador > 0,
        (trabajo[valor_col] - trabajo["promedio_movil"]).abs() / denominador,
        np.nan
    )
    
    # 5. Marcar como anomalía
    trabajo["es_anomalia"] = (
        trabajo["promedio_movil"].notna() & 
        (trabajo["desviacion_relativa"] > umbral)
    )
    
    # 6 y 7. Devolver el DataFrame (ya está ordenado y con índice reiniciado)
    return trabajo
