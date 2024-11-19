# General-purpose libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Statistical analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Data preprocessing and splitting
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Machine Learning models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

# Evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Feature selection
from mrmr import mrmr_regression

# Saving models
from joblib import dump

# Custom utilities
from energy_consumption_architecture.utils.paths import data_dir
##########################################################################################################
# Función para estandarizar los datos
def standardize_data(data):
    """
    Estandariza las características numéricas de un DataFrame.

    Parámetros:
    - data: DataFrame con características a estandarizar.

    Retorna:
    - DataFrame estandarizado.
    """
    numeric_columns = data.select_dtypes(include=["number"]).columns
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data[numeric_columns]), columns=numeric_columns)
    return data_scaled


##########################################################################################################

class MRMRSelector(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para la selección de características utilizando MRMR.
    """
    def __init__(self, target, n_features):
        self.target = target
        self.n_features = n_features
        self.selected_features = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self.selected_features = mrmr_regression(X=X, y=y, K=self.n_features)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.selected_features]

    def get_selected_features(self):
        return self.selected_features


def build_pipeline(model, requires_scaling):
    """
    Construye un pipeline que incluye selección de características basada en importancia y el modelo final.

    Parámetros:
    - model: Modelo de regresión.
    - requires_scaling: Booleano indicando si el modelo requiere estandarización.

    Retorna:
    - pipeline: Pipeline completo.
    """
    steps = []

    # Añadir escalado si el modelo lo requiere
    if requires_scaling:
        steps.append(("scaler", StandardScaler()))

    # Selección de características basada en un modelo base (RandomForest)
    feature_selector = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=42),
        threshold="median"  # Selección basada en la importancia relativa
    )
    steps.append(("feature_selector", feature_selector))

    # Añadir el modelo final
    steps.append(("regressor", model))

    # Construir el pipeline
    pipeline = Pipeline(steps=steps)
    return pipeline






##########################################################################################################
def extract_time_features(df):
    """
    Extrae características temporales de un índice de tipo datetime en un DataFrame.

    Parámetros:
    - df: DataFrame con un índice datetime.

    Retorna:
    - df: DataFrame con nuevas columnas para características temporales.
    """
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df
def time_series_train_test_split(df, target, test_size=0.2):
    """
    Divide un DataFrame en conjuntos de entrenamiento y prueba respetando el orden temporal.

    Parámetros:
    - df: DataFrame con los datos.
    - target: Nombre de la columna objetivo.
    - test_size: Proporción de datos para el conjunto de prueba.

    Retorna:
    - X_train, X_test: Conjuntos de características para entrenamiento y prueba.
    - y_train, y_test: Conjuntos objetivo para entrenamiento y prueba.
    """
    n_test = int(len(df) * test_size)
    train = df[:-n_test]
    test = df[-n_test:]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]
    return X_train, X_test, y_train, y_test

##########################################################################################################

def determine_optimal_features(X, y, max_features=None, model=None, cv=3, scoring='neg_root_mean_squared_error'):
    """
    Determina automáticamente el número óptimo de características para MRMR.

    Parámetros:
    - X: DataFrame de características.
    - y: Variable objetivo.
    - max_features: Número máximo de características a probar.
    - model: Modelo a utilizar para evaluación (por defecto RandomForestRegressor).
    - cv: Número de particiones para validación cruzada.
    - scoring: Métrica para evaluación del modelo.

    Retorna:
    - best_k: Número óptimo de características.
    - performance_history: Lista con los rendimientos para cada número de características probadas.
    """
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    if max_features is None:
        max_features = len(X.columns)

    performance_history = []
    for k in range(1, max_features + 1):
        selected_features = mrmr_regression(X=X, y=y, K=k)
        X_selected = X[selected_features]

        scores = cross_val_score(model, X_selected, y, cv=cv, scoring=scoring)
        performance_history.append(-scores.mean())

    best_k = np.argmin(performance_history) + 1
    return best_k, performance_history


##########################################################################################################

def detect_overfitting(metrics_df, threshold_ratio=1.5):
    """
    Detecta modelos sobreentrenados basándose en la relación entre el RMSE de prueba y entrenamiento.

    Parámetros:
    - metrics_df: DataFrame con métricas de los modelos.
    - threshold_ratio: Relación máxima permitida entre Test RMSE y Train RMSE.

    Retorna:
    - overfitted_models: Lista de nombres de modelos considerados como sobreentrenados.
    """
    overfitted_models = []

    for _, row in metrics_df.iterrows():
        train_rmse = row["Train RMSE"]
        test_rmse = row["Test RMSE"]
        model_name = row["Model Name"]

        if train_rmse == 0:
            if test_rmse > threshold_ratio:
                overfitted_models.append(model_name)
        else:
            test_to_train_ratio = test_rmse / train_rmse
            if test_to_train_ratio > threshold_ratio:
                overfitted_models.append(model_name)

    return overfitted_models
def select_best_model(metrics_df, overfitted_models, criteria=["Test RMSE", "Test MAE", "Test R2"]):
    """
    Selecciona el mejor modelo basado en métricas y excluyendo modelos sobreentrenados.

    Parámetros:
    - metrics_df: DataFrame con métricas de los modelos.
    - overfitted_models: Lista de nombres de modelos considerados como sobreentrenados.
    - criteria: Lista de columnas usadas para ordenar los modelos.

    Retorna:
    - best_model: Serie con las métricas del mejor modelo.
    """
    filtered_metrics = metrics_df[~metrics_df["Model Name"].isin(overfitted_models)]

    if filtered_metrics.empty:
        raise ValueError("All models were detected as overfitted. No suitable model available.")

    ascending_order = [True if col in ["Test RMSE", "Test MAE"] else False for col in criteria]
    sorted_metrics = filtered_metrics.sort_values(by=criteria, ascending=ascending_order)

    return sorted_metrics.iloc[0]


##########################################################################################################
def select_important_features(X, y, threshold=0.15):
    """
    Selecciona las características más importantes basándose en la correlación absoluta con la variable objetivo.

    Parámetros:
    - X: DataFrame con características.
    - y: Serie o array con la variable objetivo.
    - threshold: Umbral mínimo de correlación absoluta para seleccionar características.

    Retorna:
    - important_features: Lista de nombres de las características seleccionadas.
    """
    correlations = X.corrwith(y).abs()
    important_features = correlations[correlations >= threshold].index.tolist()
    return important_features

##########################################################################################################

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa un modelo de regresión.

    Parámetros:
    - model: Modelo de regresión.
    - X_train, X_test: Conjuntos de características para entrenamiento y prueba.
    - y_train, y_test: Conjuntos objetivo para entrenamiento y prueba.

    Retorna:
    - metrics: Diccionario con métricas de evaluación.
    """
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Train MAE": mean_absolute_error(y_train, y_pred_train),
        "Train R2": r2_score(y_train, y_pred_train),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Test MAE": mean_absolute_error(y_test, y_pred_test),
        "Test R2": r2_score(y_test, y_pred_test)
    }

    return metrics


def regression_pipeline_for_clusters(data, target, models, test_size=0.2, threshold_ratio=1.5):
    """
    Ejecuta el pipeline de regresión para cada cluster de series de tiempo, generando pipelines completos.

    Parámetros:
    - data: DataFrame con series de tiempo promedio por cluster.
    - target: Nombre de la variable objetivo.
    - models: Diccionario de modelos de regresión.
    - test_size: Proporción del conjunto de prueba.
    - threshold_ratio: Relación máxima para detectar sobreentrenamiento.

    Retorna:
    - metrics_by_cluster: DataFrame con métricas de todos los modelos por cluster.
    - best_pipelines: Diccionario con el mejor pipeline entrenado para cada cluster.
    """
    metrics_by_cluster = []
    best_pipelines = {}

    for cluster in data['Cluster'].unique():
        print(f"Processing Cluster {cluster}")

        # Filtrar los datos del cluster actual
        cluster_data = data[data['Cluster'] == cluster].copy()

        # Extraer características temporales
        cluster_data = extract_time_features(cluster_data)

        # Remover el índice de tiempo temporalmente
        cluster_data = cluster_data.reset_index(drop=True)

        # Eliminar la columna 'Cluster' del conjunto de datos
        cluster_data = cluster_data.drop(columns=['Cluster'])

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = time_series_train_test_split(cluster_data, target, test_size)

        cluster_metrics = []
        best_score = float('-inf')
        best_pipeline = None

        for model_name, model in models.items():
            # Determinar si el modelo requiere estandarización
            requires_scaling = isinstance(model, (LinearRegression, SVR, XGBRegressor))

            # Construir el pipeline con selección de características basada en importancia
            pipeline = build_pipeline(model, requires_scaling=requires_scaling)
            print()
            # Entrenar el pipeline
            pipeline.fit(X_train, y_train)

            # Evaluar el pipeline
            metrics = train_and_evaluate_model(pipeline, X_train, X_test, y_train, y_test)
            metrics["Model Name"] = model_name
            metrics["Cluster"] = cluster
            cluster_metrics.append(metrics)

            # Actualizar el mejor pipeline basado en Test R²
            if metrics["Test R2"] > best_score:
                best_score = metrics["Test R2"]
                best_pipeline = pipeline

        # Detectar modelos sobreentrenados y seleccionar el mejor modelo
        overfitted_models = detect_overfitting(pd.DataFrame(cluster_metrics), threshold_ratio)
        best_model_metrics = select_best_model(pd.DataFrame(cluster_metrics), overfitted_models)

        # Almacenar el mejor pipeline para el cluster
        best_pipelines[cluster] = best_pipeline

        # Agregar métricas de todos los modelos a la lista general
        metrics_by_cluster.extend(cluster_metrics)

    # Convertir las métricas a DataFrame
    metrics_df = pd.DataFrame(metrics_by_cluster)

    return metrics_df, best_pipelines
##########################################################################################################
from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calcula RMSE y sMAPE entre valores reales y predichos.

    Parámetros:
    - y_true: Serie con los valores reales.
    - y_pred: Serie con los valores predichos.

    Retorna:
    - rmse: Error cuadrático medio (RMSE).
    - smape: Porcentaje de error absoluto simétrico medio (sMAPE).
    """
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # sMAPE
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

    return rmse, smape
def evaluate_all_clusters(data, pipelines, target):
    """
    Evalúa todas las series de tiempo de cada cluster utilizando los pipelines entrenados.

    Parámetros:
    - data: DataFrame con todas las series de tiempo, etiquetas de cluster e identificadores de series.
    - pipelines: Diccionario con los mejores pipelines por cluster.
    - target: Nombre de la variable objetivo.

    Retorna:
    - result: Diccionario con métricas promedio por cluster.
    """
    result = {}

    for cluster, pipeline in pipelines.items():
        print(f"Evaluating Cluster {cluster}")
        
        # Filtrar datos del cluster
        cluster_data = data[data['Cluster'] == cluster].copy()
        
        # Inicializar métricas para el cluster
        cluster_rmse = []
        cluster_smape = []

        for series_id in cluster_data['series_id'].unique():
            # Filtrar datos de la serie actual
            series_data = cluster_data[cluster_data['series_id'] == series_id].copy()

            # Extraer características temporales
            series_data = extract_time_features(series_data)

            # Separar X (características) e y (variable objetivo)
            X = series_data.drop(columns=[target, 'Cluster', 'series_id'])
            y = series_data[target]

            # Realizar predicciones
            y_pred = pipeline.predict(X)

            # Calcular métricas
            rmse, smape = calculate_metrics(y, y_pred)

            # Guardar métricas de la serie
            cluster_rmse.append(rmse)
            cluster_smape.append(smape)

        # Calcular métricas promedio para el cluster
        avg_rmse_cluster = np.mean(cluster_rmse)
        avg_smape_cluster = np.mean(cluster_smape)

        # Guardar métricas promedio del cluster
        result[f"Cluster {cluster}"] = {
            "Average RMSE": avg_rmse_cluster,
            "Average sMAPE (%)": avg_smape_cluster
        }

    return result



