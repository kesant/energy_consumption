from energy_consumption_architecture.utils.paths import data_raw_dir,data_dir
import pandas as pd
import numpy as np
import os
import re
def load_all_series(files, columns_to_keep=None):
    """
    Carga y procesa todas las series de tiempo desde múltiples archivos.

    Parámetros:
    - files: Lista de nombres de archivos a cargar.
    - columns_to_keep: Lista de columnas a mantener en cada archivo. Si es None, se cargan todas las columnas.

    Retorna:
    - combined_df: DataFrame combinado con todas las series de tiempo y un identificador único de serie.

    Excepciones:
    - ValueError: Si la lista de archivos está vacía.
    - FileNotFoundError: Si algún archivo de la lista no existe.
    """
    if not files:
        raise ValueError("La lista de archivos está vacía. Proporcione al menos un archivo.")

    dfs = []  # Lista para almacenar cada DataFrame cargado

    # Cargar y procesar cada archivo
    for i, file in enumerate(files):
        # Generar la ruta del archivo utilizando data_dir
        file_path = data_dir("raw", file)
        
        # Verificar si el archivo existe
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        
        # Cargar todas las columnas si columns_to_keep es None
        if columns_to_keep:
            df = pd.read_csv(file_path, usecols=columns_to_keep)
        else:
            df = pd.read_csv(file_path)
        
        # Extraer el nombre base del archivo sin extensión y asignarlo como nombre del archivo
        file_name = file_path.stem
        df["file_name"] = file_name

        # Agregar un identificador de serie único
        df["series_id"] = f"series_{i + 1}"

        # Añadir el DataFrame procesado a la lista
        dfs.append(df)

    # Combinar todos los DataFrames en uno solo
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    return combined_df
