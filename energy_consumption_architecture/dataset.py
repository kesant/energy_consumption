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
    """
    dfs = []  # Lista para almacenar cada DataFrame cargado

    # Cargar y procesar cada archivo
    for i, file in enumerate(files):
        file_route = data_dir("raw", file)
        
        # Cargar todas las columnas si columns_to_keep es None
        if columns_to_keep:
            df = pd.read_csv(file_route, usecols=columns_to_keep)
        else:
            df = pd.read_csv(file_route)
        
        # Procesar la columna de fecha y tiempo
        df["Date/Time"] = '2004 ' + df["Date/Time"]
        date_format = '%Y %m/%d %H:%M:%S'
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], format=date_format, errors='coerce')
        
        # Extraer el tipo de edificio del nombre del archivo
        match = re.match(r'^[^_]+', file)
        name = match.group(0) if match else f"building_{i + 1}"
        df["type_building"] = name

        # Agregar un identificador de serie único
        df["series_id"] = f"series_{i + 1}"

        # Añadir el DataFrame procesado a la lista
        dfs.append(df)

    # Combinar todos los DataFrames en uno solo
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    return combined_df
