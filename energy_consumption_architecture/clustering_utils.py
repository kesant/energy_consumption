import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score, silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from itertools import product
from energy_consumption_architecture.utils.paths import data_dir
#######################################################################################
def build_pipeline(pca_model=None, clustering_model=None):
    """
    Construye un pipeline con estandarización, opcionalmente PCA y un modelo de clustering.

    Parámetros:
    - pca_model: Modelo PCA ajustado (o None si no se utiliza PCA).
    - clustering_model: Modelo de clustering ajustado.

    Retorna:
    - pipeline: Objeto Pipeline entrenado.
    """
    steps = [("scaler", StandardScaler())]

    if pca_model:
        steps.append(("pca", pca_model))

    if clustering_model:
        steps.append(("clustering", clustering_model))

    return Pipeline(steps)
#######################################################################################

def calculate_average_time_series_by_cluster(data_complete, data):
    """
    Esta función combina `data_complete` con etiquetas de cluster en `data` y calcula la serie de tiempo promedio
    para cada cluster.

    Parámetros:
    - data_complete: DataFrame con la columna `series_id` y la serie de tiempo completa.
    - data: DataFrame con las columnas `series_id` y `Cluster` que contiene las etiquetas de cluster.

    Retorna:
    - average_time_series_by_cluster: DataFrame con la serie de tiempo promedio por cluster.
    """

    # Hacer un merge para agregar la etiqueta de cluster al DataFrame de series concatenadas
    combined_df_with_clusters = data_complete.merge(data[['series_id', 'Cluster']], on='series_id', how='left')

    # Agrupar por cluster y fecha/hora, y calcular el promedio de cada serie de tiempo
    average_time_series_by_cluster = combined_df_with_clusters.groupby(['Cluster', 'Date/Time']).mean(numeric_only=True)
    average_time_series_by_cluster = average_time_series_by_cluster.reset_index()
    
    # Convertir la columna de fecha a formato datetime y establecerla como índice
    average_time_series_by_cluster["Date/Time"] = pd.to_datetime(average_time_series_by_cluster["Date/Time"])
    average_time_series_by_cluster.set_index('Date/Time', inplace=True)
    
    return average_time_series_by_cluster
# from kneebow.rotor import Rotor
def calculate_statistics(df):
    """
    Calcula la media y la desviación estándar de cada columna numérica en un DataFrame de series de tiempo.

    Parámetros:
    - df: DataFrame con las series de tiempo cargadas.

    Retorna:
    - df_stats: DataFrame con las estadísticas (media y desviación estándar) de cada columna numérica.
    """
    resultados = []  # Lista para almacenar las estadísticas

    # Calcular estadísticas para cada serie de tiempo única en 'series_id'
    for series_id in df["series_id"].unique():
        subset = df[df["series_id"] == series_id]
        stats = {"series_id": series_id}
        
        # Seleccionar solo las columnas numéricas
        numeric_columns = subset.select_dtypes(include=[np.number]).columns
        
        # Calcular estadísticas solo para columnas numéricas
        for column in numeric_columns:
            stats[column + '_mean'] = subset[column].mean()
            stats[column + '_std_dev'] = subset[column].std()
        
        resultados.append(stats)

    # Convertir la lista de estadísticas en un DataFrame
    df_stats = pd.DataFrame(resultados)

    return df_stats
#######################################################################################

def optimal_k_selection(X, max_k=10):
    """
    Calcula el número óptimo de clusters usando el índice de silueta y el método del codo.

    Parámetros:
    - X: Dataset (matriz de características).
    - max_k: Número máximo de clusters a evaluar. Por defecto es 10.

    Retorna:
    - optimal_k: Número óptimo de clusters seleccionado.
    """

    Sum_of_squared_distances = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        y = km.fit_predict(X)
        Sum_of_squared_distances.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X, y))

    # Determinación del número óptimo de clusters según el índice de silueta
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]

    # Determinación del número óptimo de clusters según el método del codo
    inertia_differences = np.diff(Sum_of_squared_distances)
    optimal_k_elbow = K_range[np.argmin(inertia_differences) + 1]  # +1 para ajustar el índice

    # Selección de un solo valor de K
    if optimal_k_silhouette == optimal_k_elbow:
        optimal_k = optimal_k_silhouette
    else:
        optimal_k = optimal_k_silhouette  # En caso de diferencia, priorizamos el índice de silueta

    return optimal_k

def optimal_dbscan_params(features, eps_range=(0.05, 0.2, 0.05), min_samples_range=(3, 12)):
    """
    Encuentra los parámetros óptimos para DBSCAN (eps y min_samples) basados en el índice de silueta.

    Parámetros:
    - features: Matriz de características para clustering.
    - eps_range: Tupla con rango de valores de eps (inicio, fin, paso).
    - min_samples_range: Tupla con valores de min_samples (inicio, fin).

    Retorna:
    - Mejor combinación de (eps, min_samples) basada en el índice de silueta.
    """

    # Paso 1: Gráfica de distancia de vecinos para estimación inicial de `eps`
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(features)
    distances, _ = neighbors_fit.kneighbors(features)

    # Paso 2: Pruebas de diferentes combinaciones de eps y min_samples
    eps_values = np.arange(*eps_range)
    min_samples_values = np.arange(*min_samples_range)
    dbscan_params = list(product(eps_values, min_samples_values))
    
    best_params = (None, None)
    best_sil_score = -1  # Inicializamos con un valor muy bajo

    # Almacenamos métricas para análisis adicional
    results = {
        'Eps': [],
        'Min_samples': [],
        'Silhouette Score': [],
        'Clusters': []
    }

    for eps, min_samples in dbscan_params:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(features)

        # Solo evaluamos si hay más de un cluster
        if len(set(labels)) > 1:
            try:
                sil_score = silhouette_score(features, labels)
            except ValueError:
                sil_score = 0  # Si no se puede calcular, asignamos 0
        else:
            sil_score = 0

        # Guardamos resultados en el diccionario
        results['Eps'].append(eps)
        results['Min_samples'].append(min_samples)
        results['Silhouette Score'].append(sil_score)
        results['Clusters'].append(len(set(labels)))

        # Actualizar los mejores parámetros si encontramos un mejor índice de silueta
        if sil_score > best_sil_score:
            best_sil_score = sil_score
            best_params = (eps, min_samples)

    # Convertimos resultados en DataFrame para análisis
    df_results = pd.DataFrame(results)

    # Resultados de pivot para visualización opcional
    pivot_sil_score = pd.pivot_table(df_results, values='Silhouette Score', columns='Eps', index='Min_samples')
    pivot_clusters = pd.pivot_table(df_results, values='Clusters', columns='Eps', index='Min_samples')

    return best_params
def optimal_clusters_hierarchical(features, method='ward', last_n=10):
    """
    Calcula el número óptimo de clusters para clustering jerárquico usando la aceleración en la linkage matrix.

    Parámetros:
    - features: Matriz de características para clustering.
    - method: Método de linkage. Por defecto es 'ward'.
    - last_n: Número de fusiones a considerar para calcular el número óptimo de clusters. Por defecto es 10.

    Retorna:
    - Número óptimo de clusters.
    """

    # Calcular la linkage matrix
    mergings = linkage(features, method=method)

    # Obtener las alturas de los últimos 'last_n' clusters
    last = mergings[-last_n:, 2]
    last_rev = last[::-1]

    # Calcular la aceleración (segunda derivada)
    acceleration = np.diff(last, 2)  # Segunda derivada de las alturas
    acceleration_rev = acceleration[::-1]

    # Encontrar el número óptimo de clusters
    optimal_k = acceleration_rev.argmax() + 2  # +2 porque se pierde una posición en cada derivada

    return optimal_k
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

# Función para ejecutar K-Means y calcular métricas
def kmeans_clustering(X, max_k=10):
    optimal_k = optimal_k_selection(X, max_k=max_k)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(X)
    num_clusters = kmeans.n_clusters
    silhouette = silhouette_score(X, clusters)
    davies_bouldin = davies_bouldin_score(X, clusters)
    return clusters, silhouette, davies_bouldin, "K-Means",num_clusters

# Función para ejecutar DBSCAN y calcular métricas
def dbscan_clustering(X, eps_range=(0.05, 0.2, 0.05), min_samples_range=(3, 12)):
    eps_min_samples = optimal_dbscan_params(X, eps_range=eps_range, min_samples_range=min_samples_range)
    dbscan = DBSCAN(eps=eps_min_samples[0], min_samples=eps_min_samples[1])
    clusters = dbscan.fit_predict(X)
    num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    if len(set(clusters)) > 1:
        silhouette = silhouette_score(X, clusters)
    else:
        silhouette = -1  # Silhouette no es aplicable si hay un solo cluster
    davies_bouldin = davies_bouldin_score(X, clusters)
    return clusters, silhouette, davies_bouldin, "DBSCAN",num_clusters

# Función para ejecutar Clustering Jerárquico y calcular métricas
def hierarchical_clustering(X, last_n=10):
    optimal_k = optimal_clusters_hierarchical(X, method='ward', last_n=last_n)
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    clusters = hierarchical.fit_predict(X)
    num_clusters = hierarchical.n_clusters
    silhouette = silhouette_score(X, clusters)
    davies_bouldin = davies_bouldin_score(X, clusters)
    return clusters, silhouette, davies_bouldin, "Hierarchical",num_clusters

#######################################################################################

# Función para determinar el número óptimo de componentes principales

def determine_optimal_pca_components(data, variance_threshold=0.95):
    """
    Determina el número óptimo de componentes principales basado en la varianza explicada acumulativa.

    Retorna:
    - optimal_components: Número óptimo de componentes.
    """
    pca = PCA()
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    return optimal_components

def apply_pca(data, variance_threshold=0.95):
    """
    Aplica PCA para reducir la dimensionalidad de los datos.

    Retorna:
    - DataFrame con componentes principales.
    - Modelo PCA ajustado.
    """
    optimal_components = determine_optimal_pca_components(data, variance_threshold)
    pca = PCA(n_components=optimal_components)
    pca_data = pca.fit_transform(data)
    pca_df = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(optimal_components)])
    return pca_df, pca
#######################################################################################

def evaluate_models(models_metrics):
    """
    Evalúa los modelos basándose en las métricas de silueta y Davies-Bouldin.

    Parámetros:
    - models_metrics: Lista de métricas de cada modelo.

    Retorna:
    - metrics: DataFrame con métricas normalizadas y combinadas.
    - best_model_info: Información del mejor modelo.
    """
    # Convertir las métricas a DataFrame
    metrics = pd.DataFrame(models_metrics, columns=["Model", "Silhouette Score", "Davies-Bouldin Index", "Num Clusters", "PCA Applied"])

    # Normalizar métricas
    metrics["Silhouette Score Norm"] = (metrics["Silhouette Score"] - metrics["Silhouette Score"].min()) / (metrics["Silhouette Score"].max() - metrics["Silhouette Score"].min())
    metrics["Davies-Bouldin Index Norm"] = (metrics["Davies-Bouldin Index"].max() - metrics["Davies-Bouldin Index"]) / (metrics["Davies-Bouldin Index"].max() - metrics["Davies-Bouldin Index"].min())

    # Calcular puntuación combinada
    metrics["Combined Score"] = metrics[["Silhouette Score Norm", "Davies-Bouldin Index Norm"]].mean(axis=1)

    # Seleccionar el mejor modelo basado en el puntaje combinado
    best_model_info = metrics.loc[metrics["Combined Score"].idxmax()].to_dict()

    return metrics, best_model_info

def run_clustering_methods(data, max_k=10, eps_range=(0.05, 0.2, 0.05), min_samples_range=(3, 12), pca_applied=False):
    """
    Ejecuta múltiples métodos de clustering y calcula métricas.

    Parámetros:
    - data: DataFrame con características para clustering.
    - max_k: Número máximo de clusters para K-Means y Clustering Jerárquico.
    - eps_range: Rango de valores para `eps` en DBSCAN.
    - min_samples_range: Rango de valores para `min_samples` en DBSCAN.
    - pca_applied: Booleano indicando si se usó PCA.

    Retorna:
    - metrics_df: DataFrame con métricas de cada modelo.
    - clusters: Etiquetas de cluster por método.
    """
    clusters_results = {}
    models_metrics = []

    # K-Means
    clusters_kmeans, silhouette_kmeans, davies_bouldin_kmeans, model_kmeans, ncluster_kmean = kmeans_clustering(data, max_k)
    clusters_results["K-Means"] = clusters_kmeans
    models_metrics.append([model_kmeans, silhouette_kmeans, davies_bouldin_kmeans, ncluster_kmean, pca_applied])

    # DBSCAN
    clusters_dbscan, silhouette_dbscan, davies_bouldin_dbscan, model_dbscan, ncluster_dbscan = dbscan_clustering(data, eps_range, min_samples_range)
    clusters_results["DBSCAN"] = clusters_dbscan
    models_metrics.append([model_dbscan, silhouette_dbscan, davies_bouldin_dbscan, ncluster_dbscan, pca_applied])

    # Hierarchical Clustering
    clusters_hierarchical, silhouette_hierarchical, davies_bouldin_hierarchical, model_hierarchical, ncluster_hierarchical = hierarchical_clustering(data)
    clusters_results["Hierarchical"] = clusters_hierarchical
    models_metrics.append([model_hierarchical, silhouette_hierarchical, davies_bouldin_hierarchical, ncluster_hierarchical, pca_applied])

    return models_metrics, clusters_results



def automated_clustering_pipeline(data, variance_threshold=0.95, max_k=10, eps_range=(0.05, 0.2, 0.05), min_samples_range=(3, 12)):
    """
    Ejecuta un pipeline completo de clustering con y sin PCA y selecciona el mejor proceso.

    Retorna:
    - metrics_combined: DataFrame con métricas de cada método evaluado.
    - best_pipeline: Objeto Pipeline entrenado del mejor proceso.
    - clustered_data: DataFrame con etiquetas de cluster del mejor proceso.
    """
    # Estandarizar los datos
    data_scaled = standardize_data(data)

    # Clustering sin PCA
    metrics_no_pca, clusters_results_no_pca = run_clustering_methods(data_scaled, max_k, eps_range, min_samples_range, pca_applied=False)

    # Clustering con PCA
    pca_data, pca_model = apply_pca(data_scaled, variance_threshold)
    metrics_pca, clusters_results_pca = run_clustering_methods(pca_data, max_k, eps_range, min_samples_range, pca_applied=True)

    # Combinar métricas de ambos procesos
    all_metrics = metrics_no_pca + metrics_pca
    metrics_combined, best_model_global = evaluate_models(all_metrics)

    # Seleccionar el mejor modelo global
    best_model_name = best_model_global["Model"]
    pca_applied = best_model_global["PCA Applied"]

    # Construir el pipeline y asignar etiquetas de cluster
    clustered_data = data.copy()

    if not pca_applied:
        clustering_model = None
        if best_model_name == "K-Means":
            clustering_model = KMeans(n_clusters=best_model_global["Num Clusters"], random_state=42).fit(data_scaled)
            clustered_data["Cluster"] = clustering_model.labels_
        elif best_model_name == "DBSCAN":
            eps_min_samples = optimal_dbscan_params(data_scaled, eps_range=eps_range, min_samples_range=min_samples_range)
            clustering_model = DBSCAN(eps=eps_min_samples[0], min_samples=eps_min_samples[1]).fit(data_scaled)
            clustered_data["Cluster"] = clustering_model.labels_
        elif best_model_name == "Hierarchical":
            clustering_model = AgglomerativeClustering(n_clusters=int(best_model_global["Num Clusters"])).fit(data_scaled)
            clustered_data["Cluster"] = clustering_model.labels_

        best_pipeline = build_pipeline(pca_model=None, clustering_model=clustering_model)

    else:
        clustering_model = None
        if best_model_name == "K-Means":
            clustering_model = KMeans(n_clusters=best_model_global["Num Clusters"], random_state=42).fit(pca_data)
            clustered_data["Cluster"] = clustering_model.labels_
        elif best_model_name == "DBSCAN":
            eps_min_samples = optimal_dbscan_params(pca_data, eps_range=eps_range, min_samples_range=min_samples_range)
            clustering_model = DBSCAN(eps=eps_min_samples[0], min_samples=eps_min_samples[1]).fit(pca_data)
            clustered_data["Cluster"] = clustering_model.labels_
        elif best_model_name == "Hierarchical":
            clustering_model = AgglomerativeClustering(n_clusters=int(best_model_global["Num Clusters"])).fit(pca_data)
            clustered_data["Cluster"] = clustering_model.labels_

        best_pipeline = build_pipeline(pca_model=pca_model, clustering_model=clustering_model)

    return metrics_combined, best_pipeline, clustered_data



