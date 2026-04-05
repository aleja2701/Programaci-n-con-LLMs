import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generar_caso_de_uso_escalar_variables_numericas():
    n_samples = np.random.randint(20, 100)
    
    # Generar DataFrame aleatorio mixto
    df_input = pd.DataFrame({
        'edad': np.random.randint(18, 70, n_samples),
        'ingresos': np.random.uniform(1000, 5000, n_samples),
        'categoria': np.random.choice(['A', 'B', 'C'], n_samples),
        'score': np.random.randn(n_samples) * 10
    })
    
    # Lógica esperada
    df_num = df_input.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_num)
    expected_output = pd.DataFrame(scaled_data, columns=df_num.columns, index=df_num.index)
    
    return {
        "input": {"df": df_input},
        "output": expected_output
    }
