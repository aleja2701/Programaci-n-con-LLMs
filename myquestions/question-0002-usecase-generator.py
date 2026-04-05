import pandas as pd
import numpy as np

def generar_caso_de_uso_extraer_features_temporales():
    start_date = pd.to_datetime('2023-01-01')
    n_samples = np.random.randint(10, 50)
    
    random_days = np.random.randint(0, 365, n_samples)
    fechas = start_date + pd.to_timedelta(random_days, unit='d')
    
    df_input = pd.DataFrame({
        'fecha_str': fechas.astype(str),
        'valor': np.random.uniform(10, 100, n_samples)
    })
    
    # Lógica esperada
    df_output = df_input.copy()
    df_output['fecha_temp'] = pd.to_datetime(df_output['fecha_str'])
    df_output['mes'] = df_output['fecha_temp'].dt.month
    df_output['dia_semana'] = df_output['fecha_temp'].dt.dayofweek
    df_output['es_fin_de_semana'] = df_output['dia_semana'].isin([5, 6]).astype(int)
    df_output = df_output.drop(columns=['fecha_str', 'fecha_temp'])
    
    return {
        "input": {"df": df_input, "col_fecha": 'fecha_str'},
        "output": df_output
    }
