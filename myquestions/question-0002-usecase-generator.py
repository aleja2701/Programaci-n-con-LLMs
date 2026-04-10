import pandas as pd
import numpy as np

def generar_caso_de_uso_extraer_features_temporales():
    # Fechas aleatorias
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2024-12-31')
    n_samples = np.random.randint(20, 100)

    random_days = np.random.randint(0, (end_date - start_date).days, n_samples)
    fechas = start_date + pd.to_timedelta(random_days, unit='d')

    df_input = pd.DataFrame({
        'fecha_str': fechas.astype(str),
        'ventas': np.random.uniform(100, 1000, n_samples)
    })

    # Output esperado
    df_output = df_input.copy()
    df_output['fecha_temp'] = pd.to_datetime(df_output['fecha_str'])
    df_output['mes'] = df_output['fecha_temp'].dt.month
    df_output['dia_semana'] = df_output['fecha_temp'].dt.dayofweek
    df_output['es_fin_de_semana'] = df_output['dia_semana'].isin([5, 6]).astype(int)
    df_output = df_output.drop(columns=['fecha_str', 'fecha_temp'])


    return {"df": df_input.copy(), "col_fecha": 'fecha_str'}, df_output
